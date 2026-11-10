"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Radar, Send } from "lucide-react";
import { useState, type FormEvent } from "react";
import { algorithms, api, friendly, json, number, stamp, type Drift, type DriftFeature, type Event, type Model, type Prediction } from "./api";
import { Chart, Grid, Notice, Panel } from "./common";
import { Button } from "./ui/button";

function DriftResults({ drift }: { drift: Drift }) {
  const [feature, choose] = useState("");
  const selected: DriftFeature | undefined = drift.features.find((x) => x.feature === feature) ?? drift.features[0];
  const relative = (counts: Record<string, number>, key: string) => {
    const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
    return total ? Math.round(1000 * (counts[key] ?? 0) / total) / 10 : 0;
  };
  return <Panel title="Drift report" hint={`${drift.rows} batch rows · ${drift.features.filter((x) => x.flagged).length} flagged features`}>
    <Grid headers={["Feature", "Method", "Statistic", "Adjusted p", "Missing shift", "Flag"]} rows={drift.features.map((x) => [
      x.feature, x.method, number(x.statistic), x.adjusted_p_value == null ? "—" : number(x.adjusted_p_value),
      `${number(100 * (x.current_missing - x.baseline_missing))} pp`, x.flagged ? "Review" : "—",
    ])} />
    {selected && <label>Feature to chart<select value={selected.feature} onChange={(e) => choose(e.target.value)}>
      {drift.features.map((x) => <option key={x.feature}>{x.feature}</option>)}
    </select></label>}
    {selected && Object.keys(selected.baseline).length > 0 && <Chart
      data={Object.keys(selected.baseline).map((label) => ({
        label, "Training %": relative(selected.baseline, label), "Batch %": relative(selected.current, label),
      }))} label="label" fields={["Training %", "Batch %"]} description={`Distribution shift for ${selected.feature}`} />}
  </Panel>;
}

function ModelTools({ model }: { model: Model }) {
  const client = useQueryClient();
  const [mode, setMode] = useState<"batch-predict" | "drift" | "predict">("batch-predict");
  const [result, setResult] = useState<Prediction>();
  const [drift, setDrift] = useState<Drift>();
  const startRows = JSON.stringify({ rows: [Object.fromEntries(Object.keys(model.metadata.schema).map((key) => [key, null]))] }, null, 2);
  const operation = useMutation({
    mutationFn: async ({ action, form }: { action: typeof mode; form: FormData }) => {
      const url = `/models/${model.id}/${action}`;
      if (action === "drift") return { drift: await api<Drift>(url, { method: "POST", body: form }) };
      let init: RequestInit = { method: "POST", body: form };
      if (action === "predict") {
        try { init = json(JSON.parse(String(form.get("rows")))); }
        catch { throw new Error("Enter a valid JSON prediction request."); }
      }
      return { prediction: await api<Prediction>(url, init) };
    },
    onSuccess: (response) => {
      if (response.drift) setDrift(response.drift);
      if (response.prediction) setResult(response.prediction);
      void client.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setDrift(undefined);
    setResult(undefined);
    operation.mutate({ action: mode, form: new FormData(event.currentTarget) });
  }
  function download() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "modelforge-predictions.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <>
    <Panel title="Inference and monitoring" hint="Evaluate a labeled batch, compare data drift, or make an online prediction.">
      <form className="stack" onSubmit={submit}>
        <label>Operation<select value={mode} disabled={operation.isPending} onChange={(e) => {
          setMode(e.target.value as typeof mode); operation.reset(); setResult(undefined); setDrift(undefined);
        }}>
          <option value="batch-predict">Batch prediction</option><option value="drift">Compare drift</option><option value="predict">Online prediction</option>
        </select></label>
        {mode === "predict" ? <label>Prediction JSON<textarea name="rows" rows={9} defaultValue={startRows} required /></label>
          : <label>Batch CSV<input name="file" type="file" accept=".csv" required /></label>}
        <Button type="submit" disabled={operation.isPending}>{mode === "drift" ? <Radar size={16} /> : <Send size={16} />}{operation.isPending ? "Processing…" : mode === "drift" ? "Analyze drift" : "Run prediction"}</Button>
        <Notice error={operation.error} />
      </form>
    </Panel>
    {result && <Panel title="Prediction results" hint={`${result.predictions.length} rows · ${number(result.latency_ms)} ms · first 20 shown`}>
      {result.performance && <div className="score-grid">{Object.entries(result.performance).map(([key, value]) => <div key={key}><span>{friendly(key)}</span><strong>{number(value)}</strong></div>)}</div>}
      <Grid headers={["Row", "Prediction", "Class probabilities"]} rows={result.predictions.slice(0, 20).map((value, index) => [
        index + 1, typeof value === "number" ? number(value) : value,
        result.probabilities?.[index] ? Object.entries(result.probabilities[index]).map(([label, probability]) => `${label}: ${number(probability)}`).join(" · ") : "—",
      ])} />
      <Button variant="outline" onClick={download}>Download JSON</Button>
    </Panel>}
    {drift && <DriftResults drift={drift} />}
  </>;
}

function MonitoringHistory({ model }: { model: Model }) {
  const history = useQuery({
    queryKey: ["activity", model.id],
    queryFn: () => api<Event[]>(`/activity?resource_id=${model.id}&limit=50`),
  });
  const predictions = (history.data ?? []).filter((event) => typeof event.details.latency_ms === "number");
  const quality = predictions.filter((event) => event.details.performance);
  return <Panel title="Monitoring history" hint="Recent requests for this version. Quality scores require a target column in the uploaded batch.">
    <Notice error={history.error} />
    {history.isLoading && <p role="status">Loading monitoring events…</p>}
    {predictions.length ? <>
      <Chart data={predictions.slice(0, 12).reverse().map((event) => ({
        request: `#${event.id}`, "Latency (ms)": Math.round(Number(event.details.latency_ms)),
      }))} label="request" fields={["Latency (ms)"]} description="Inference latency across recent requests" />
      {!!quality.length && <Grid headers={["Labeled batch", "Rows", "Observed performance"]} rows={quality.map((event) => [
        stamp(event.created_at), String(event.details.rows), Object.entries(event.details.performance as Record<string, number>)
          .map(([key, value]) => `${friendly(key)}: ${number(value)}`).join(" · "),
      ])} />}
    </> : <p className="empty">Run a prediction to start recording latency.</p>}
  </Panel>;
}

export function Models({ models }: { models: Model[] }) {
  const [selected, select] = useState("");
  const model = models.find((x) => x.id === selected) ?? models[0];
  return <>
    <Panel title="Model registry" hint="Successful experiments register immutable, checksummed versions.">
      {model ? <>
        <label>Model version<select value={model.id} onChange={(e) => select(e.target.value)}>
          {models.map((x) => <option key={x.id} value={x.id}>{x.name} · v{x.version}</option>)}
        </select></label>
        <Grid headers={["Algorithm", "Dataset version", "Registered", "Artifact"]} rows={[[algorithms[model.metadata.config.algorithm], model.metadata.dataset_version, stamp(model.created_at), <code key={model.id}>{model.artifact_key}</code>]]} />
        <details><summary>Feature schema and library versions</summary>
          <Grid headers={["Feature", "Type"]} rows={Object.entries(model.metadata.schema)} />
          <Grid headers={["Library", "Version"]} rows={Object.entries(model.metadata.libraries)} />
          <p className="muted"><code>{model.metadata.sha256}</code></p>
        </details>
      </> : <p className="empty">Train an experiment to register a model.</p>}
    </Panel>
    {model && <><ModelTools key={model.id} model={model} /><MonitoringHistory model={model} /></>}
  </>;
}
