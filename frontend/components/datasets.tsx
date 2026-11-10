"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { useState, type FormEvent } from "react";
import { api, number, type Dataset, type Profile } from "./api";
import { Chart, Grid, Notice, Panel } from "./common";
import { Button } from "./ui/button";

export function DatasetWorkspace({ datasets }: { datasets: Dataset[] }) {
  const client = useQueryClient();
  const [selected, select] = useState("");
  const [feature, setFeature] = useState("");
  const current = selected || datasets[0]?.id || "";
  const profile = useQuery({
    queryKey: ["profile", current], queryFn: () => api<Profile>(`/datasets/${current}/profile`),
    enabled: !!current,
  });
  const upload = useMutation({
    mutationFn: (body: FormData) => api<Dataset>("/datasets", { method: "POST", body }),
    onSuccess: (data) => {
      select(data.id);
      setFeature("");
      void client.invalidateQueries({ queryKey: ["datasets"] });
      void client.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  const field = profile.data?.columns.find((x) => x.name === feature) ?? profile.data?.columns[0];
  const correlations = Object.keys(profile.data?.correlations ?? {});
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    upload.mutate(new FormData(event.currentTarget));
  }
  return <>
    <div className="two">
      <Panel title="Upload a dataset" hint="Use a UTF-8 CSV. Reuse a name to create a new version.">
        <form className="stack" onSubmit={submit}>
          <label>Dataset name<input name="name" maxLength={120} placeholder="e.g. Iris benchmark" required /></label>
          <label>CSV file<input name="file" type="file" accept=".csv" required /></label>
          <Button type="submit" disabled={upload.isPending}><Upload size={16} />{upload.isPending ? "Uploading…" : "Upload CSV"}</Button>
          <Notice error={upload.error} />
        </form>
      </Panel>
      <Panel title="Dataset versions" hint="Each version preserves the uploaded source and its checksum.">
        {datasets.length ? <label>Dataset to inspect<select value={current} onChange={(e) => { select(e.target.value); setFeature(""); }}>
          {datasets.map((x) => <option key={x.id} value={x.id}>{x.name} · v{x.version} · {number(x.rows)} rows</option>)}
        </select></label> : <p className="empty">Upload a CSV to start.</p>}
      </Panel>
    </div>
    <Notice error={profile.error} />
    {profile.isLoading && <p role="status">Profiling dataset…</p>}
    {profile.data && <>
      <Panel title="Data quality" hint={`${number(profile.data.rows)} rows across ${profile.data.columns.length} columns`}>
        <Grid headers={["Column", "Inferred type", "Missing", "Distinct"]} rows={profile.data.columns.map((x) =>
          [x.name, x.dtype, number(x.missing), number(x.unique)])} />
      </Panel>
      {field && <Panel title="Feature distribution" hint="Numeric histograms use ten bins. Categorical charts show the ten most frequent values.">
        <label>Feature<select value={field.name} onChange={(e) => setFeature(e.target.value)}>
          {profile.data.columns.map((x) => <option key={x.name}>{x.name}</option>)}
        </select></label>
        {field.distribution.length ? <Chart data={field.distribution} label="label" fields={["count"]} description={`Distribution of ${field.name}`} /> : <p className="empty">All values are missing.</p>}
        {!!Object.keys(field.statistics).length && <Grid headers={["Statistic", "Value"]} rows={Object.entries(field.statistics).map(([key, value]) => [key, number(value)])} />}
      </Panel>}
      <Panel title="Sample rows" hint="First 12 source records">
        <Grid headers={profile.data.columns.map((x) => x.name)} rows={profile.data.preview.map((row) =>
          profile.data!.columns.map((x) => row[x.name] == null ? "—" : String(row[x.name])))} />
      </Panel>
      {!!correlations.length && <Panel title="Pearson correlation" hint="Pairwise correlations for numeric columns">
        <Grid headers={["Feature", ...correlations]} rows={correlations.map((a) => [
          a, ...correlations.map((b) => {
            const value = profile.data!.correlations[a][b];
            return value == null ? "—" : number(value);
          }),
        ])} />
      </Panel>}
    </>}
  </>;
}
