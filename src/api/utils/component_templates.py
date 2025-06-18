"""
Component templates for generating v0 UI specs from ComfyDeploy deployments.
"""

from typing import List
import re


def _slugify_simple(text: str) -> str:
    """Simple slugify helper (keeps lower-case ascii, replaces others with -)"""
    text = re.sub(r"[^a-zA-Z0-9\-\_]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-").lower()


def generate_component_code(inputs: List[dict]) -> tuple[str, List[str]]:
    """Create a minimal ShadCN/New-York page component string for the given inputs.
    Returns (component_source, registry_dependencies)."""

    control_snippets: List[str] = []
    deps: set[str] = {"input", "label", "button"}

    for inp in inputs:
        label = inp.get("display_name") or inp.get("input_id") or inp.get("class_type")
        field_name = inp.get("input_id") or _slugify_simple(label)
        input_type = inp.get("type")
        default_value = inp.get("default_value")

        if input_type in {"float", "integer"}:
            dv_raw = default_value if default_value is not None else ""
            dv = str(dv_raw)
            snippet = (
                f'<div className="flex flex-col gap-2">\n'
                f'  <Label htmlFor="{field_name}">{label}</Label>\n'
                f'  <Input type="number" id="{field_name}" name="{field_name}" defaultValue="{dv}" />\n'
                f'</div>'
            )
        elif input_type == "boolean":
            deps.add("checkbox")
            checked_value = str(default_value).lower() if default_value is not None else "false"
            checkbox_line = (
                f'  <Checkbox id="{field_name}" name="{field_name}" defaultChecked={{' + checked_value + '}} />\n'
            )
            snippet = (
                '<div className="flex items-center gap-2">\n'
                + checkbox_line
                + f'  <Label htmlFor="{field_name}">{label}</Label>\n'
                + '</div>'
            )
        elif inp.get("enum_values"):
            # Enumerated string options -> Select component
            deps.update({"select"})
            options = inp["enum_values"] or []
            options_code = "\n".join(
                [f'        <SelectItem value="{opt}">{opt}</SelectItem>' for opt in options]
            )
            dv = default_value if default_value is not None else (options[0] if options else "")
            snippet = (
                f'<div className="flex flex-col gap-2">\n'
                f'  <Label htmlFor="{field_name}">{label}</Label>\n'
                f'  <Select defaultValue="{dv}" name="{field_name}">\n'
                f'    <SelectTrigger>{label}</SelectTrigger>\n'
                f'    <SelectContent>\n'
                f'{options_code}\n'
                f'    </SelectContent>\n'
                f'  </Select>\n'
                f'</div>'
            )
        else:
            # Default to text input
            dv_raw = default_value if default_value is not None else ""
            dv = str(dv_raw).replace('"', '\\"')
            snippet = (
                f'<div className="flex flex-col gap-2">\n'
                f'  <Label htmlFor="{field_name}">{label}</Label>\n'
                f'  <Input id="{field_name}" name="{field_name}" placeholder="{label}" defaultValue="{dv}" />\n'
                f'</div>'
            )

        control_snippets.append(snippet)

    controls_code = "\n      ".join(control_snippets)

    # Include card dependency always for layout
    deps.add("card")

    # Build individual import lines for each component group so that paths match registry files
    import_lines: List[str] = [
        'import type React from "react"',
        'import { useState, useEffect, useRef } from "react"',
        'import { Input } from "@/components/ui/input"',
        'import { Label } from "@/components/ui/label"',
        'import { Button } from "@/components/ui/button"',
        'import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card"',
    ]
    if "checkbox" in deps:
        import_lines.append('import { Checkbox } from "@/components/ui/checkbox"')
    if "select" in deps:
        import_lines.append('import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select"')
    import_lines.append('import { Loader2, CheckCircle, XCircle, Hourglass } from "lucide-react"')

    imports = "\n".join(import_lines)

    # Build the component using the new useEffect-based polling approach
    component_template = '''
"use client"

__IMPORTS__

type PollData = {
  live_status?: string
  status?: "queued" | "running" | "success" | "failed" | "api_error"
  outputs?: Array<{ url?: string; [key: string]: any }>
  progress?: number
  queue_position?: number | null
  error?: any
  details?: any
}

function WorkflowForm() {
  const [runId, setRunId] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [mutationError, setMutationError] = useState<string | null>(null)

  const [pollingData, setPollingData] = useState<PollData | null>(null)
  const [isPolling, setIsPolling] = useState<boolean>(false)
  const [pollingError, setPollingError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState<boolean>(false)

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Effect for polling
  useEffect(() => {
    const clearPollingInterval = () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
      setIsPolling(false)
    }

    if (!runId) {
      clearPollingInterval()
      setPollingData(null)
      return
    }

    const fetchAndPoll = async () => {
      if (!runId) return

      setIsPolling(true)
      setPollingError(null)

      try {
        const response = await fetch(`/api/poll?runId=${runId}`)
        const data: PollData = await response.json()

        if (!response.ok) {
          const errorMsg = data.error || `Poll API Error: ${response.status}`
          setPollingError(errorMsg)
          setPollingData({ ...data, status: "api_error", live_status: errorMsg })
        } else {
          setPollingData(data)
        }

        // Check if polling should stop based on the new data
        if (data.status === "success" || data.status === "failed") {
          clearPollingInterval()
        }
      } catch (err: any) {
        setPollingError(err.message || "Polling failed unexpectedly.")
      }
    }

    // Initial fetch then set interval
    fetchAndPoll()

    clearPollingInterval() // Clear any existing interval before starting a new one

    pollIntervalRef.current = setInterval(async () => {
      await fetchAndPoll()
    }, 2000) // Poll every 2 seconds

    return () => {
      clearPollingInterval()
    }
  }, [runId])

  // Effect to update image URL from pollingData
  useEffect(() => {
    if (pollingData?.status === "success") {
      const output = pollingData.outputs?.[0]
      if (output?.url) {
        setImageUrl(output.url)
      }
    }
  }, [pollingData])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const inputs = Object.fromEntries(formData.entries())

    // Clear old run state
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
    setRunId(null)
    setImageUrl(null)
    setMutationError(null)
    setPollingData(null)
    setIsPolling(false)
    setPollingError(null)
    setIsGenerating(true)

    try {
      setMutationError(null)
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      })
      const responseData = await res.json()
      
      if (!res.ok) {
        const errorMsg = responseData.error || `API Error: ${res.status}`
        const errorDetails = responseData.details ? JSON.stringify(responseData.details) : "No details"
        throw new Error(`${errorMsg} - ${errorDetails}`)
      }

      if (responseData && typeof responseData.run_id === "string" && responseData.run_id.length > 0) {
        setRunId(responseData.run_id)
        setImageUrl(null)
        setPollingData(null)
        setPollingError(null)
      } else {
        setMutationError(`Failed to start run: run_id missing. Response: ${JSON.stringify(responseData)}`)
        setRunId(null)
      }
    } catch (error: any) {
      setMutationError(error.message)
    } finally {
      setIsGenerating(false)
    }
  }

  const displayStatus = pollingData?.status
  const overallIsLoading =
    isGenerating || (isPolling && !!runId && displayStatus !== "success" && displayStatus !== "failed")

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-8">
      <Card className="w-full max-w-xl border shadow-sm">
        <CardHeader>
          <CardTitle className="text-2xl">ComfyDeploy Workflow</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
            __CONTROLS__
            <div className="flex justify-end">
              <Button type="submit" size="sm" disabled={isGenerating}>
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  "Run Workflow"
                )}
              </Button>
            </div>
          </form>

          {mutationError && (
            <div className="mt-4 text-center text-sm font-medium text-red-600">Error: {mutationError}</div>
          )}

          {(overallIsLoading || displayStatus) && !mutationError && (
            <div className="mt-6 flex flex-col items-center gap-2">
              {overallIsLoading &&
                !displayStatus && (
                  <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>{isGenerating ? "Queuing run..." : "Initializing poll..."}</span>
                  </div>
                )}
              {displayStatus && (
                <div className="flex items-center justify-center gap-2 text-sm capitalize text-muted-foreground">
                  {displayStatus === "queued" && <Hourglass className="h-4 w-4 animate-pulse text-amber-500" />}
                  {displayStatus === "running" && <Loader2 className="h-4 w-4 animate-spin text-blue-500" />}
                  {displayStatus === "api_error" && <XCircle className="h-4 w-4 text-orange-500" />}
                  {displayStatus === "success" && <CheckCircle className="h-4 w-4 text-green-500" />}
                  {displayStatus === "failed" && <XCircle className="h-4 w-4 text-red-500" />}
                  <span>Status: {pollingData?.live_status || displayStatus}</span>
                  {displayStatus === "queued" && pollingData?.queue_position != null && (
                    <span> (Queue: {pollingData.queue_position})</span>
                  )}
                  {displayStatus === "running" && pollingData?.progress != null && (
                    <span> ({Math.round(pollingData.progress * 100)}%)</span>
                  )}
                </div>
              )}
            </div>
          )}

          {pollingError && !mutationError && (
            <div className="mt-4 text-center text-sm text-red-600">Polling Error: {pollingError}</div>
          )}

          {imageUrl && (
            <div className="mt-6 flex justify-center">
              <img
                src={imageUrl || "/placeholder.svg"}
                alt="Generated output"
                className="max-w-full rounded-md border shadow-sm"
              />
            </div>
          )}

          {pollingData && (
            <details className="mt-6 w-full">
              <summary className="cursor-pointer text-xs text-muted-foreground">View Raw Output</summary>
              <pre className="mt-2 overflow-x-auto rounded bg-muted p-4 text-xs">
                {JSON.stringify(pollingData, null, 2)}
              </pre>
            </details>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default function Page() {
  return <WorkflowForm />
}
'''

    component_code = (
        component_template
        .replace("__IMPORTS__", imports)
        .replace("__CONTROLS__", controls_code)
    )
    
    return component_code, sorted(deps)


def generate_api_routes(deployment_id: str, api_base: str) -> tuple[str, str]:
    """Generate the API route files for generate and poll endpoints.
    Returns (generate_route_content, poll_route_content)."""
    
    generate_route = f'''import {{ NextRequest, NextResponse }} from 'next/server'

export async function POST(req: NextRequest) {{
  // Inputs sent from the client form
  const inputs = await req.json();

  const res = await fetch('{api_base}/run/deployment/queue', {{
    method: 'POST',
    headers: {{
      'Content-Type': 'application/json',
      Authorization: `Bearer ${{process.env.COMFY_API_KEY ?? ''}}`,
    }},
    body: JSON.stringify({{
      deployment_id: '{deployment_id}',
      inputs: inputs,
    }}),
  }});

  const data = await res.json();
  return NextResponse.json(data);
}}
'''

    poll_route = f'''import {{ NextRequest, NextResponse }} from 'next/server'

export async function GET(req: NextRequest) {{
  const {{ searchParams }} = new URL(req.url)
  const runId = searchParams.get('runId')

  const res = await fetch('{api_base}/run/' + runId, {{
    headers: {{
      Authorization: `Bearer ${{process.env.COMFY_API_KEY ?? ''}}`,
    }},
  }})

  const json = await res.json()

  const {{ live_status, status, outputs, progress, queue_position }} = json

  // Now you can use the run_id in your response
  return NextResponse.json({{
    live_status,
    status,
    outputs,
    progress,
    queue_position,
  }})
}}
'''

    # Normalize curly braces for valid TypeScript
    generate_route = generate_route.replace("{{", "{").replace("}}", "}")
    poll_route = poll_route.replace("{{", "{").replace("}}", "}")
    
    return generate_route, poll_route 