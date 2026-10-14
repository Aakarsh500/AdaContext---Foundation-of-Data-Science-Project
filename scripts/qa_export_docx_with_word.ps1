param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPdf
)

$ErrorActionPreference = 'Stop'
$word = $null
$document = $null

try {
    Write-Output 'Starting Microsoft Word renderer...'
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.ConfirmConversions = $false
    $word.Options.SaveNormalPrompt = $false
    $word.Options.UpdateLinksAtOpen = $false

    Write-Output 'Opening DOCX...'
    $document = $word.Documents.Open($InputPath, $false, $true, $false)
    Write-Output 'Exporting PDF...'
    $document.ExportAsFixedFormat($OutputPdf, 17, $false, 0, 0, 1, 999, 0, $true, $true, 1, $true, $true, $false)
    Write-Output 'PDF export complete.'
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($null -ne $word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
