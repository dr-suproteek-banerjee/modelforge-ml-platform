"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, Play, Square } from "lucide-react";
import { useState, type FormEvent } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { active, algorithms, api, friendly, json, number, type Algorithm, type Config, type Dataset, type Experiment, type Metrics, type Task } from "./api";
import { Chart, Grid, Notice, Panel } from "./common";
import { Button } from "./ui/button";

function Wizard({ datasets, onCreate }: { datasets: Dataset[]; onCreate: (id: string) => void }) {
  const client = useQueryClient();
  const capabilities = useQuery({ queryKey: ["capabilities"], queryFn: () => api<{ torch_mlp: boolean }>("/capabilities") });
  const [step, setStep] = useState(1);
  const [datasetId, setDataset] = useState("");
  const [task, setTask] = useState<Task>("classification");
  const [algorithm, setAlgorithm] = useState<Algorithm>("logistic");
  const [name, setName] = useState("");
  const [target, setTarget] = useState("");
  const dataset = datasets.find((x) => x.id === datasetId) ?? datasets[0];
  const targetValue = dataset?.columns.includes(target) ? target : dataset?.columns.at(-1) ?? "";
  const create = useMutation({
    mutationFn: (config: Config) => api<Experiment>("/experiments", json(config)),
    onSuccess: (row) => {
      onCreate(row.id);
      void client.invalidateQueries({ queryKey: ["experiments"] });
      void client.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dataset) return;
    if (step === 1) { setStep(2); return; }
    const form = new FormData(event.currentTarget);
    create.mutate({
      name, dataset_id: dataset.id, target: targetValue, task, algorithm,
      seed: Number(form.get("seed")), test_size: Number(form.get("holdout")),
      cv_folds: Number(form.get("folds")), trials: algorithm === "linear" ? 0 : Number(form.get("trials")),
      preprocessing: {
        numeric_imputer: String(form.get("numeric")), categorical_imputer: String(form.get("categorical")),
        scale: form.has("scale"), exclude: form.getAll("exclude").map(String),
      },
    });
  }
  if (!dataset) return <p className="empty">Upload a dataset before configuring an experiment.</p>;
  return <form className="stack" onSubmit={submit}>
    <div className="eyebrow">Step {step} of 2 · {step === 1 ? "Experiment setup" : "Preprocessing and validation"}</div>
    {step === 1 ? <div className="form-grid">
      <label>Experiment name<input value={name} onChange={(e) => setName(e.target.value)} maxLength={120} required /></label>
      <label>Dataset version<select value={dataset.id} onChange={(e) => { setDataset(e.target.value); setTarget(""); }}>
        {datasets.map((x) => <option key={x.id} value={x.id}>{x.name} · v{x.version}</option>)}
      </select></label>
      <label>Target column<select value={targetValue} onChange={(e) => setTarget(e.target.value)}>
        {dataset.columns.map((x) => <option key={x}>{x}</option>)}
      </select></label>
      <label>Task<select value={task} onChange={(e) => { const value = e.target.value as Task; setTask(value); setAlgorithm(value === "regression" ? "linear" : "logistic"); }}>
        <option value="classification">Classification</option><option value="regression">Regression</option>
      </select></label>
      <label>Algorithm<select value={algorithm} onChange={(e) => setAlgorithm(e.target.value as Algorithm)}>
        {(Object.keys(algorithms) as Algorithm[]).filter((key) => key !== (task === "regression" ? "logistic" : "linear") && (key !== "torch_mlp" || capabilities.data?.torch_mlp)).map((key) => <option key={key} value={key}>{algorithms[key]}</option>)}
      </select></label>
    </div> : <div className="form-grid">
      <label>Numeric imputer<select name="numeric"><option value="median">Median</option><option value="mean">Mean</option><option value="most_frequent">Most frequent</option></select></label>
      <label>Categorical imputer<select name="categorical"><option value="most_frequent">Most frequent</option><option value="constant">Missing category</option></select></label>
      <label>Holdout fraction<input name="holdout" type="number" defaultValue="0.2" min="0.1" max="0.4" step="0.05" required /></label>
      <label>CV folds<input name="folds" type="number" defaultValue="3" min="2" max="8" required /></label>
      <label>Random seed<input name="seed" type="number" defaultValue="42" min="0" max="4294967295" required /></label>
      <label>Optuna trials (0 disables)<input name="trials" type="number" defaultValue="0" min="0" max="30" disabled={algorithm === "linear"} required /></label>
      <label className="check"><input name="scale" type="checkbox" defaultChecked /> Standardize numeric features</label>
      <label>Exclude columns (optional)<select name="exclude" multiple size={Math.min(5, dataset.columns.length)}>
        {dataset.columns.filter((x) => x !== targetValue).map((x) => <option key={x}>{x}</option>)}
      </select></label>
    </div>}
    <div className="row">
      {step === 2 && <Button variant="outline" onClick={() => setStep(1)}>Back</Button>}
      <Button type="submit" disabled={create.isPending}><FlaskConical size={16} />{step === 1 ? "Configure preprocessing" : create.isPending ? "Saving…" : "Create experiment"}</Button>
    </div>
    <Notice error={create.error} />
    {create.isSuccess && <p role="status">Experiment created. Start training in the run panel.</p>}
  </form>;
}

function Evaluation({ metrics }: { metrics: Metrics }) {
  const report = Object.entries(metrics.classification_report);
  return <>
    <Panel title="Held-out evaluation" hint="Scores from the untouched test split. Cross-validation uses the training split.">
      <div className="score-grid">{Object.entries(metrics.scores).map(([key, value]) => <div key={key}><span>{friendly(key)}</span><strong>{number(value)}</strong></div>)}</div>
      <Grid headers={["Cross-validation metric", "Value"]} rows={Object.entries(metrics.cross_validation).map(([key, value]) => [friendly(key), number(value)])} />
      {Object.keys(metrics.best_parameters).length > 0 && <details><summary>Best Optuna parameters</summary>
        <Grid headers={["Parameter", "Value"]} rows={Object.entries(metrics.best_parameters).map(([key, value]) => [key, number(value)])} />
      </details>}
    </Panel>
    {!!metrics.labels.length && <div className="two">
      <Panel title="Confusion matrix" hint="Rows are actual classes. Columns are predictions.">
        <Grid headers={["Actual / predicted", ...metrics.labels]} rows={metrics.confusion_matrix.map((row, i) => [metrics.labels[i], ...row])} />
      </Panel>
      <Panel title="Classification report" hint="Precision, recall, and F1 by class">
        <Grid headers={["Class", "Precision", "Recall", "F1", "Support"]} rows={report.map(([label, x]) =>
          [label, number(x.precision), number(x.recall), number(x["f1-score"]), x.support])} />
      </Panel>
    </div>}
    {!!metrics.residuals.length && <Panel title="Regression residuals" hint="Actual minus predicted. The horizontal axis is the predicted value.">
      <div className="chart" role="img" aria-label="Residual scatter plot">
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 420, height: 245 }}>
          <ScatterChart><CartesianGrid stroke="#263347" />
            <XAxis type="number" dataKey="prediction" name="Prediction" tick={{ fill: "#9aa9c1" }} />
            <YAxis type="number" dataKey="residual" name="Residual" tick={{ fill: "#9aa9c1" }} />
            <Tooltip /><Scatter data={metrics.residuals} fill="#7de4c1" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Panel>}
    <Panel title="Feature importance" hint="Mean change in score after permuting each original holdout feature.">
      <Chart data={metrics.feature_importance.slice(0, 12)} label="feature" fields={["importance"]} description="Permutation feature importance" />
      <details><summary>All feature scores</summary>
        <Grid headers={["Feature", "Importance"]} rows={metrics.feature_importance.map((x) => [x.feature, number(x.importance)])} />
      </details>
    </Panel>
  </>;
}

export function Experiments({ experiments, datasets }: { experiments: Experiment[]; datasets: Dataset[] }) {
  const client = useQueryClient();
  const [selected, select] = useState("");
  const id = selected || experiments[0]?.id;
  const listEntry = experiments.find((x) => x.id === id);
  const detail = useQuery({
    queryKey: ["experiment", id], queryFn: () => api<Experiment>(`/experiments/${id}`), enabled: !!id,
    refetchInterval: (query) => query.state.data && active(query.state.data) ? 900 : false,
  });
  const run = detail.data ?? listEntry;
  const trackingRun = run?.mlflow_run ?? listEntry?.mlflow_run;
  const evaluation = useQuery({
    queryKey: ["metrics", id], queryFn: () => api<Metrics>(`/experiments/${id}/metrics`), enabled: run?.status === "completed",
  });
  const action = useMutation({
    mutationFn: (event: { id: string; operation: "train" | "cancel" }) =>
      api<Experiment>(`/experiments/${event.id}/${event.operation}`, { method: "POST" }),
    onSuccess: (result) => {
      client.setQueryData(["experiment", result.id], result);
      void client.invalidateQueries({ queryKey: ["experiments"] });
      void client.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  return <>
    <Panel title="Training configuration" hint="Select a dataset and define how to fit and evaluate the model."><Wizard datasets={datasets} onCreate={select} /></Panel>
    <Panel title="Training runs" hint="Inspect progress, queue a draft, retry a failure, or cancel an active job.">
      {experiments.length > 0 && <label>Experiment<select value={id ?? ""} onChange={(e) => select(e.target.value)}>
        {experiments.map((x) => <option key={x.id} value={x.id}>{x.config.name} · {x.status} · {x.id.slice(0, 7)}</option>)}
      </select></label>}
      <Notice error={detail.error ?? action.error} />
      {run ? <div className="stack run">
        <div className="row"><strong>{run.config.name}</strong><span className={`badge ${run.status}`}>{run.status}</span>
          <span className="muted">{algorithms[run.config.algorithm]}</span></div>
        <progress aria-label="Training progress" value={run.progress} max={100} /><span>{run.progress}% complete</span>
        <Notice error={run.error} />
        {trackingRun && <span className="muted">MLflow run <code>{trackingRun}</code></span>}
        {active(run) ? <Button variant="outline" disabled={action.isPending} onClick={() => action.mutate({ id: run.id, operation: "cancel" })}><Square size={15} />Cancel training</Button>
          : run.status !== "completed" && <Button disabled={action.isPending} onClick={() => action.mutate({ id: run.id, operation: "train" })}><Play size={15} />Start training</Button>}
      </div> : <p className="empty">Create your first experiment.</p>}
    </Panel>
    <Notice error={evaluation.error} />
    {evaluation.data && <Evaluation metrics={evaluation.data} />}
  </>;
}

export function Comparison({ experiments }: { experiments: Experiment[] }) {
  const [task, setTask] = useState<Task>("classification");
  const [metric, setMetric] = useState("f1_macro");
  const [descending, setDescending] = useState(true);
  const options = task === "classification" ? ["f1_macro", "accuracy", "log_loss"] : ["rmse", "mae", "r2"];
  const values = experiments.flatMap((x) => {
    const score = x.metrics?.scores[metric];
    return x.config.task === task && score != null ? [{ x, score }] : [];
  }).sort((a, b) => (a.score - b.score) * (descending ? -1 : 1));
  return <Panel title="Compare experiments" hint="Compare runs using the same dataset version, target, seed, and holdout fraction for a fair ranking.">
    <div className="form-grid">
      <label>Task<select value={task} onChange={(e) => {
        const next = e.target.value as Task;
        setTask(next); setMetric(next === "classification" ? "f1_macro" : "rmse"); setDescending(next === "classification");
      }}><option value="classification">Classification</option><option value="regression">Regression</option></select></label>
      <label>Rank by<select value={metric} onChange={(e) => {
        setMetric(e.target.value); setDescending(!["log_loss", "rmse", "mae"].includes(e.target.value));
      }}>{options.map((x) => <option key={x} value={x}>{friendly(x)}</option>)}</select></label>
      <label>Direction<select value={descending ? "high" : "low"} onChange={(e) => setDescending(e.target.value === "high")}><option value="high">High to low</option><option value="low">Low to high</option></select></label>
    </div>
    {values.length ? <>
      <Chart data={values.slice(0, 12).map(({ x, score }) => ({ name: x.config.name, score }))} label="name" fields={["score"]} description={`Experiment comparison by ${friendly(metric)}`} />
      <Grid headers={["Experiment", "Algorithm", "Seed", friendly(metric)]} rows={values.map(({ x, score }) => [x.config.name, algorithms[x.config.algorithm], x.config.seed, number(score)])} />
    </> : <p className="empty">Complete a {task} run to populate this view.</p>}
  </Panel>;
}
