#!/usr/bin/env python3
"""
Manuscript Review Tool for Amazon KDP
Checks manuscripts for common issues before building.
"""

import argparse
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from collections import Counter


def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def check_double_spaces(text):
    """Find lines with double spaces."""
    issues = []
    for i, line in enumerate(text.split('\n'), 1):
        if '  ' in line and not line.startswith('    ') and not line.strip().startswith('#'):
            issues.append(f"  Line {i}: double spaces found")
    return issues


def check_trailing_whitespace(text):
    """Find lines with trailing whitespace."""
    issues = []
    for i, line in enumerate(text.split('\n'), 1):
        if line != line.rstrip():
            issues.append(f"  Line {i}: trailing whitespace")
    return issues


def check_inconsistent_quotes(text):
    """Check for mixed straight and curly quotes."""
    straight = text.count('"') + text.count("'")
    curly = text.count('\u201c') + text.count('\u201d') + text.count('\u2018') + text.count('\u2019')
    issues = []
    if straight > 0 and curly > 0:
        issues.append(f"  Mixed quote styles: {straight} straight, {curly} curly")
    return issues


def check_chapter_structure(text):
    """Check for consistent chapter headings."""
    issues = []
    h1_chapters = len(re.findall(r'^#\s+Chapter\s+\d+', text, re.MULTILINE | re.IGNORECASE))
    h2_chapters = len(re.findall(r'^##\s+(Chapter\s+\d+|\d+\.\s+)', text, re.MULTILINE | re.IGNORECASE))
    
    if h1_chapters == 0 and h2_chapters == 0:
        issues.append("  No recognizable chapter headings found (expected '# Chapter N' or '## Chapter N')")
    elif h1_chapters > 0 and h2_chapters > 0:
        issues.append(f"  Inconsistent chapter levels: {h1_chapters} H1, {h2_chapters} H2")
    
    return issues


def check_scene_breaks(text):
    """Check for consistent scene breaks."""
    issues = []
    asterisk_breaks = len(re.findall(r'\n\*\s*\*\s*\*\s*\n', text))
    hash_breaks = len(re.findall(r'\n#\s*#\s*#\s*\n', text))
    dash_breaks = len(re.findall(r'\n---\s*\n', text))
    
    types = sum(x > 0 for x in [asterisk_breaks, hash_breaks, dash_breaks])
    if types > 1:
        issues.append(f"  Mixed scene break styles: ***({asterisk_breaks}), ###({hash_breaks}), ---({dash_breaks})")
    return issues


def check_word_count(text):
    """Estimate word count."""
    words = len(text.split())
    status = ""
    if words < 2500:
        status = "(short story / pamphlet)"
    elif words < 15000:
        status = "(novella)"
    elif words < 40000:
        status = "(novellette)"
    else:
        status = "(novel)"
    return words, status


def check_front_matter(text):
    """Check for common front matter elements."""
    issues = []
    has_real_title = False
    for match in re.finditer(r'^#\s+(.+)$', text, re.MULTILINE):
        candidate = match.group(1).strip().lower()
        if not any(skip in candidate for skip in ['table of contents', 'contents', 'toc', 'dedication', 'acknowledgments', 'foreword', 'preface']):
            has_real_title = True
            break
    if not has_real_title:
        issues.append("  No title (H1 heading) found at start")
    return issues


def check_dialogue_formatting(text):
    """Check for common dialogue issues."""
    issues = []
    # Dialogue without closing quote before paragraph break
    bad_dialogue = re.findall(r'^"[^"\n]+\n', text, re.MULTILINE)
    if bad_dialogue:
        issues.append(f"  {len(bad_dialogue)} lines may have unclosed dialogue quotes")
    return issues


def generate_report(filepath, text):
    """Generate a full review report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"MANUSCRIPT REVIEW: {filepath}")
    lines.append("=" * 60)
    
    words, status = check_word_count(text)
    lines.append(f"\n📊 WORD COUNT: {words:,} {status}")
    
    lines.append("\n🔍 STRUCTURE CHECKS")
    lines.append("-" * 40)
    chapter_issues = check_chapter_structure(text)
    if chapter_issues:
        lines.extend(chapter_issues)
    else:
        lines.append("  ✓ Chapter structure looks consistent")
    
    scene_issues = check_scene_breaks(text)
    if scene_issues:
        lines.extend(scene_issues)
    else:
        lines.append("  ✓ Scene breaks are consistent")
    
    front_issues = check_front_matter(text)
    if front_issues:
        lines.extend(front_issues)
    else:
        lines.append("  ✓ Title/heading found")
    
    lines.append("\n✏️  FORMATTING CHECKS")
    lines.append("-" * 40)
    
    ds_issues = check_double_spaces(text)
    if ds_issues:
        lines.append(f"  ⚠️  Double spaces: {len(ds_issues)} instances")
        lines.extend(ds_issues[:5])
        if len(ds_issues) > 5:
            lines.append(f"  ... and {len(ds_issues) - 5} more")
    else:
        lines.append("  ✓ No double spaces")
    
    tw_issues = check_trailing_whitespace(text)
    if tw_issues:
        lines.append(f"  ⚠️  Trailing whitespace: {len(tw_issues)} instances")
    else:
        lines.append("  ✓ No trailing whitespace")
    
    quote_issues = check_inconsistent_quotes(text)
    if quote_issues:
        lines.extend(quote_issues)
    else:
        lines.append("  ✓ Quote style is consistent")
    
    dialogue_issues = check_dialogue_formatting(text)
    if dialogue_issues:
        lines.extend(dialogue_issues)
    else:
        lines.append("  ✓ Dialogue formatting looks okay")
    
    # Character frequency for common issues
    lines.append("\n📈 CHARACTER STATS")
    lines.append("-" * 40)
    non_ascii = [c for c in set(text) if ord(c) > 127]
    if non_ascii:
        lines.append(f"  Non-ASCII characters found: {len(non_ascii)}")
        counts = Counter(c for c in text if ord(c) > 127)
        for char, count in counts.most_common(10):
            lines.append(f"    '{char}' (U+{ord(char):04X}): {count}")
    else:
        lines.append("  ✓ All ASCII characters")
    
    lines.append("\n" + "=" * 60)
    lines.append("REVIEW COMPLETE")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Review a manuscript for KDP production issues')
    parser.add_argument('manuscript', help='Path to the manuscript markdown file')
    parser.add_argument('-o', '--output', help='Output report file (default: prints to stdout)')
    parser.add_argument('--market', action='store_true', help='Also run market check (title, keywords, Amazon branding)')
    args = parser.parse_args()
    
    filepath = Path(args.manuscript)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    text = read_file(filepath)
    report = generate_report(filepath, text)
    
    if args.market:
        script_dir = Path(__file__).parent
        market_script = script_dir / 'market_check.py'
        if market_script.exists():
            result = subprocess.run(
                [sys.executable, str(market_script), str(filepath)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                report += "\n\n" + result.stdout
            else:
                report += "\n\n[Market check could not be completed]\n"
        else:
            report += "\n\n[Market check script not found]\n"
    
    if args.output:
        outpath = Path(args.output)
        # Append date to filename to prevent overwrites
        date_stamp = datetime.now().strftime("%Y-%m-%d")
        stem = outpath.stem
        suffix = outpath.suffix
        dated_path = outpath.with_name(f"{stem}_{date_stamp}{suffix}")
        dated_path.write_text(report, encoding='utf-8')
        print(f"Review saved to: {dated_path}")
    else:
        print(report)


if __name__ == '__main__':
    main()
