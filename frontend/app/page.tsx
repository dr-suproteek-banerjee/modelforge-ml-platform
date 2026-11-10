"use client";

import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Activity, Boxes, Database, FlaskConical, GitCompareArrows, Layers3, LayoutGrid } from "lucide-react";
import { useState } from "react";
import { active, algorithms, api, friendly, number, stamp, type Dataset, type Event, type Experiment, type Model } from "../components/api";
import { Grid, Notice, Panel } from "../components/common";
import { DatasetWorkspace } from "../components/datasets";
import { Comparison, Experiments } from "../components/experiments";
import { Models } from "../components/models";
import { Button } from "../components/ui/button";

const pages = [
  { key: "overview", label: "Overview", icon: LayoutGrid, title: "Experiment overview", description: "Track your datasets, training runs, and registered models in one workspace." },
  { key: "datasets", label: "Datasets", icon: Database, title: "Dataset manager", description: "Version CSV datasets and inspect their quality before training." },
  { key: "experiments", label: "Experiments", icon: FlaskConical, title: "Training experiments", description: "Build a reproducible pipeline and evaluate it on a held-out split." },
  { key: "compare", label: "Compare", icon: GitCompareArrows, title: "Compare runs", description: "Explore held-out scores and rank completed experiments." },
  { key: "models", label: "Models", icon: Boxes, title: "Model registry", description: "Use versioned models for inference and monitor changes in incoming data." },
  { key: "activity", label: "Activity", icon: Activity, title: "Workspace activity", description: "Review uploads, training lifecycle events, inference, and drift checks." },
] as const;
type Page = typeof pages[number]["key"];

function Dashboard() {
  const [page, setPage] = useState<Page>("overview");
  const descriptor = pages.find((x) => x.key === page)!;
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => api<Dataset[]>("/datasets") });
  const runs = useQuery({ queryKey: ["experiments"], queryFn: () => api<Experiment[]>("/experiments"), refetchInterval: 2500 });
  const models = useQuery({ queryKey: ["models"], queryFn: () => api<Model[]>("/models"), refetchInterval: page === "models" || page === "overview" ? 3500 : false });
  const events = useQuery({ queryKey: ["activity"], queryFn: () => api<Event[]>("/activity"), enabled: page === "activity", refetchInterval: 3500 });
  const loading = datasets.isPending || runs.isPending || models.isPending;
  const error = datasets.error ?? runs.error ?? models.error;
  function navigate(key: Page) { setPage(key); window.scrollTo({ top: 0 }); }
  const summaries = [
    { label: "Dataset versions", value: datasets.data?.length ?? 0, icon: Database, to: "datasets" },
    { label: "Experiments", value: runs.data?.length ?? 0, icon: FlaskConical, to: "experiments" },
    { label: "Active jobs", value: runs.data?.filter(active).length ?? 0, icon: Activity, to: "experiments" },
    { label: "Registered versions", value: models.data?.length ?? 0, icon: Boxes, to: "models" },
  ] as const;
  return <div className="shell">
    <aside className="rail">
      <div className="brand"><span><Layers3 size={21} /></span><div>ModelForge<small>ML WORKSPACE</small></div></div>
      <nav aria-label="Workspace"><div className="rail-label">Workspace</div>{pages.map(({ key, label, icon: Icon }) =>
        <button key={key} type="button" aria-current={page === key ? "page" : undefined} onClick={() => navigate(key)}><Icon size={18} />{label}</button>)}</nav>
      <div className="rail-foot"><span /> Local workspace</div>
    </aside>
    <main>
      <header className="hero"><p className="eyebrow">Model lifecycle / {descriptor.label}</p><h1>{descriptor.title}</h1><p>{descriptor.description}</p></header>
      <div className="stack content">
        <Notice error={error} />
        {loading && <p role="status">Loading workspace…</p>}
        {page === "overview" && <>
          <div className="overview-grid">{summaries.map(({ label, value, icon: Icon, to }) => <button className="summary" key={label} onClick={() => navigate(to)}><Icon size={18} /><span>{label}</span><strong>{loading ? "—" : number(value)}</strong></button>)}</div>
          <Panel title="Recent experiments" hint="The latest runs across all datasets.">
            {runs.data?.length ? <Grid headers={["Experiment", "Algorithm", "Status", "Created"]} rows={runs.data.slice(0, 8).map((x) => [x.config.name, algorithms[x.config.algorithm], <span key={x.id} className={`badge ${x.status}`}>{x.status}</span>, stamp(x.created_at)])} /> :
              <p className="empty">Your workspace is ready. Upload a dataset to begin.</p>}
            <div className="row"><Button onClick={() => navigate("datasets")}>Open datasets</Button><Button variant="outline" onClick={() => navigate("experiments")}>Open experiments</Button></div>
          </Panel>
        </>}
        {page === "datasets" && <DatasetWorkspace datasets={datasets.data ?? []} />}
        {page === "experiments" && <Experiments datasets={datasets.data ?? []} experiments={runs.data ?? []} />}
        {page === "compare" && <Comparison experiments={runs.data ?? []} />}
        {page === "models" && <Models models={models.data ?? []} />}
        {page === "activity" && <Panel title="Activity log" hint="Latest events are listed first.">
          <Notice error={events.error} />
          {events.isLoading ? <p role="status">Loading activity…</p> : events.data?.length ? <Grid
            headers={["Event", "Resource", "Details", "Time"]} rows={events.data.map((event) => [
              friendly(event.action), <code key={event.id}>{event.resource_id}</code>,
              JSON.stringify(event.details), stamp(event.created_at),
            ])} /> : <p className="empty">No activity yet.</p>}
        </Panel>}
      </div>
    </main>
  </div>;
}

export default function Page() {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 1000 } } }));
  return <QueryClientProvider client={client}><Dashboard /></QueryClientProvider>;
}
