# Finish the Word version of the report: called by tools/latex_to_word.py.
# Opens the generated HTML in Microsoft Word, saves it as .docx and applies the university layout:
# A4, Times New Roman 12 pt, 1.5 line spacing, 3.5 cm binding margin, heading styles, table of contents,
# lists of figures and tables, roman page numbers for the front matter and arabic numbers from Chapter 1.
param([string]$Html, [string]$Docx)
$ErrorActionPreference = 'Stop'

function B($v) { if ($v) { -1 } else { 0 } }     # Word's True is -1

function Set-Style($doc, $name, [hashtable]$p) {
    try { $s = $doc.Styles.Item($name) } catch { return }
    $f = $s.Font
    $f.Name = 'Times New Roman'
    if ($p.ContainsKey('font')) { $f.Name = $p.font }
    if ($p.ContainsKey('size')) { $f.Size = [single]$p.size }
    if ($p.ContainsKey('bold')) { $f.Bold = (B $p.bold) }
    if ($p.ContainsKey('italic')) { $f.Italic = (B $p.italic) }
    $f.Color = -16777216
    if ($s.Type -ne 1) { return }          # character styles stop here
    $pf = $s.ParagraphFormat
    if ($p.ContainsKey('align')) { $pf.Alignment = $p.align }
    if ($p.ContainsKey('before')) { $pf.SpaceBefore = [single]$p.before }
    if ($p.ContainsKey('after')) { $pf.SpaceAfter = [single]$p.after }
    if ($p.ContainsKey('single')) { if ($p.single) { $pf.LineSpacingRule = 0 } else { $pf.LineSpacingRule = 1 } }
    if ($p.ContainsKey('keep')) { $pf.KeepWithNext = (B $p.keep) }
    if ($p.ContainsKey('pagebreak')) { $pf.PageBreakBefore = (B $p.pagebreak) }
    if ($p.ContainsKey('left')) { $pf.LeftIndent = [single]$p.left }
    if ($p.ContainsKey('first')) { $pf.FirstLineIndent = [single]$p.first }
    if ($p.ContainsKey('outline')) { $pf.OutlineLevel = $p.outline }
}

function Replace-WithField($doc, $text, $code) {
    $r = $doc.Content
    $f = $r.Find
    $f.ClearFormatting()
    if ($f.Execute($text)) {
        $r.Text = ''
        $null = $doc.Fields.Add($r, -1, $code, $false)
    } else { Write-Output "placeholder not found: $text" }
}

$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $d = $w.Documents.Open($Html, $false, $false, $false)
    $d.SaveAs2($Docx, 16)
    $d.Close(0)
    $d = $w.Documents.Open($Docx)
    $w.ActiveWindow.View.Type = 3

    # ---- pictures: embed them in the file instead of linking to the PNGs
    $n = 0
    foreach ($s in @($d.InlineShapes)) {
        if ($s.LinkFormat -ne $null) { $s.LinkFormat.SavePictureWithDocument = $true; $s.LinkFormat.BreakLink(); $n++ }
    }
    Write-Output "pictures embedded: $n of $($d.InlineShapes.Count)"

    # ---- page set-up: A4, 2.5 cm margins with a 3.5 cm binding margin
    foreach ($sec in $d.Sections) {
        $ps = $sec.PageSetup
        $ps.PageWidth = 595.3; $ps.PageHeight = 841.9
        $ps.TopMargin = 70.9; $ps.BottomMargin = 70.9; $ps.LeftMargin = 99.2; $ps.RightMargin = 70.9
        $ps.HeaderDistance = 35.4; $ps.FooterDistance = 35.4
    }

    # ---- styles
    Set-Style $d (-1)  @{ size = 12; align = 3; before = 0; after = 6; single = $false }            # Normal
    Set-Style $d (-2)  @{ size = 16; bold = $true; align = 1; before = 0; after = 18; single = $true; keep = $true; pagebreak = $true }
    Set-Style $d (-3)  @{ size = 14; bold = $true; align = 0; before = 18; after = 6; single = $true; keep = $true }
    Set-Style $d (-4)  @{ size = 12; bold = $true; align = 0; before = 12; after = 6; single = $true; keep = $true }
    Set-Style $d (-5)  @{ size = 12; bold = $true; italic = $true; align = 0; before = 10; after = 4; single = $true; keep = $true }
    Set-Style $d (-20) @{ size = 12; bold = $true; before = 6; after = 2; single = $true; align = 0 }   # TOC 1
    Set-Style $d (-21) @{ size = 12; before = 0; after = 2; single = $true; left = 18; align = 0 }       # TOC 2
    Set-Style $d (-22) @{ size = 11; before = 0; after = 1; single = $true; left = 36; align = 0 }       # TOC 3
    Set-Style $d (-32) @{ size = 10; italic = $true }                                               # Header
    Set-Style $d (-33) @{ size = 11 }                                                               # Footer
    Set-Style $d 'figurecaption'  @{ size = 11; align = 1; before = 2; after = 12; single = $true }
    Set-Style $d 'tablecaption'   @{ size = 11; align = 1; before = 10; after = 4; single = $true; keep = $true }
    Set-Style $d 'listingcaption' @{ size = 11; align = 1; before = 10; after = 4; single = $true; keep = $true }
    Set-Style $d 'figureimage'    @{ size = 11; align = 1; before = 6; after = 2; single = $true; keep = $true }
    Set-Style $d 'equation'       @{ align = 1; before = 4; after = 6; single = $true }
    Set-Style $d 'tabletext'      @{ size = 10; align = 0; before = 0; after = 0; single = $true }
    Set-Style $d 'tablenote'      @{ size = 10; align = 0; before = 3; after = 10; single = $true }
    Set-Style $d 'labelitem'      @{ size = 12; align = 3; before = 0; after = 3; left = 36; first = -36; single = $false }
    Set-Style $d 'reference'      @{ size = 11; align = 0; before = 0; after = 4; left = 18; first = -18; single = $true }
    Set-Style $d 'codeblock'      @{ font = 'Consolas'; size = 8.5; align = 0; before = 0; after = 0; single = $true }
    Set-Style $d 'codeinline'     @{ font = 'Consolas'; size = 10.5 }
    Set-Style $d 'frontheading'   @{ size = 16; bold = $true; align = 1; before = 0; after = 18; single = $true; pagebreak = $true }
    Set-Style $d 'titleline'      @{ align = 1; before = 0; after = 0; single = $true }
    Set-Style $d 'centered'       @{ align = 1 }
    try { $d.Styles.Item('codeblock').ParagraphFormat.Shading.BackgroundPatternColor = 15921906 } catch {}

    # ---- tables: centred, repeated header row, no row split
    foreach ($t in $d.Tables) {
        $t.Rows.Alignment = 1
        if ($t.Range.InlineShapes.Count -gt 0) {
            $t.Borders.Enable = $false
        } else {
            try { $t.Rows.Item(1).HeadingFormat = -1 } catch {}
            $t.Rows.AllowBreakAcrossPages = 0
        }
    }

    # ---- table of contents, list of figures, list of tables
    Replace-WithField $d '[[tableofcontents]]' 'TOC \o "1-3" \h \z \u'
    # the lists use level 9 so that their entries get the plain "TOC 9" style instead of bold "TOC 1"
    Replace-WithField $d '[[listoffigures]]' 'TOC \h \z \t "figurecaption,9"'
    Replace-WithField $d '[[listoftables]]' 'TOC \h \z \t "tablecaption,9"'
    Set-Style $d (-28) @{ size = 11; bold = $false; before = 0; after = 3; single = $true; left = 0; align = 0 }

    # ---- section break before Chapter 1: roman page numbers before it, arabic from it
    $r = $d.Content
    $f = $r.Find
    $f.ClearFormatting()
    $f.Style = $d.Styles.Item(-2).NameLocal
    if ($f.Execute('Chapter 1:')) {
        $r.Collapse(1)
        $r.InsertBreak(2)
        # the empty paragraph that now holds the section break must not start a page of its own
        $last = $d.Sections.Item(1).Range.Paragraphs.Last
        if ($last.Range.Text.Length -le 2) {
            $last.Style = $d.Styles.Item(-1).NameLocal
            $last.PageBreakBefore = 0
        }
    } else { Write-Output 'Chapter 1 heading not found' }

    $s1 = $d.Sections.Item(1)
    $s1.PageSetup.DifferentFirstPageHeaderFooter = -1
    $fr = $s1.Footers.Item(1).Range
    $fr.ParagraphFormat.Alignment = 1
    $null = $d.Fields.Add($fr, 33)
    $s1.Footers.Item(1).PageNumbers.NumberStyle = 2
    $s1.Footers.Item(1).PageNumbers.RestartNumberingAtSection = $true
    $s1.Footers.Item(1).PageNumbers.StartingNumber = 1

    if ($d.Sections.Count -ge 2) {
        $s2 = $d.Sections.Item(2)
        $s2.PageSetup.DifferentFirstPageHeaderFooter = 0
        $s2.Footers.Item(1).LinkToPrevious = $false
        $s2.Headers.Item(1).LinkToPrevious = $false
        $s2.Footers.Item(1).PageNumbers.NumberStyle = 0
        $s2.Footers.Item(1).PageNumbers.RestartNumberingAtSection = $true
        $s2.Footers.Item(1).PageNumbers.StartingNumber = 1
        $hr = $s2.Headers.Item(1).Range
        $hr.Text = ''
        $null = $d.Fields.Add($hr, -1, 'STYLEREF "Heading 1"', $false)
        $hb = $s2.Headers.Item(1).Range.ParagraphFormat.Borders.Item(-3)
        $hb.LineStyle = 1; $hb.LineWidth = 4
        # the first paragraph of the new section already starts a page
        $s2.Range.Paragraphs.Item(1).PageBreakBefore = 0
    }

    $d.Fields.Update() | Out-Null
    foreach ($toc in $d.TablesOfContents) { $toc.Update() }
    $d.Repaginate()
    foreach ($toc in $d.TablesOfContents) { $toc.UpdatePageNumbers() }
    Write-Output "pages: $($d.ComputeStatistics(2))  tables: $($d.Tables.Count)  pictures: $($d.InlineShapes.Count)  TOCs: $($d.TablesOfContents.Count)"
    $d.Save()
    $pdf = [System.IO.Path]::ChangeExtension($Html, '.pdf')
    $d.ExportAsFixedFormat($pdf, 17)
    Write-Output "PDF preview: $pdf"
    $d.Close(0)
} catch {
    Write-Output "ERROR: $($_.Exception.Message)"
    Write-Output $_.InvocationInfo.PositionMessage
    exit 1
} finally {
    foreach ($x in @($w.Documents)) { $x.Close(0) }
    $w.Quit()
}
