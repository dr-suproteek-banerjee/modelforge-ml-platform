export type Cell = string | number | boolean | null;
export type Dataset = { id: string; name: string; version: number; rows: number; columns: string[]; sha256: string; created_at: string };
export type Column = { name: string; dtype: string; missing: number; unique: number; statistics: Record<string, number>; distribution: { label: string; count: number }[] };
export type Profile = { rows: number; columns: Column[]; preview: Record<string, Cell>[]; correlations: Record<string, Record<string, number | null>> };
export type Metrics = {
  scores: Record<string, number>;
  cross_validation: Record<string, number>;
  labels: string[];
  confusion_matrix: number[][];
  classification_report: Record<string, Record<string, number>>;
  residuals: { actual: number; prediction: number; residual: number }[];
  feature_importance: { feature: string; importance: number }[];
  best_parameters: Record<string, number>;
};
export type Task = "classification" | "regression";
export type Algorithm = "logistic" | "linear" | "random_forest" | "gradient_boosting" | "xgboost" | "torch_mlp";
export type Config = {
  name: string; dataset_id: string; target: string; task: Task; algorithm: Algorithm; seed: number;
  test_size: number; cv_folds: number; trials: number;
  preprocessing: { numeric_imputer: string; categorical_imputer: string; scale: boolean; exclude: string[] };
};
export type Experiment = { id: string; config: Config; status: "draft" | "queued" | "running" | "completed" | "failed" | "cancelled"; progress: number; error: string | null; metrics: Metrics | null; mlflow_run: string | null; created_at: string };
export type Model = { id: string; experiment_id: string; name: string; version: number; artifact_key: string; metadata: { schema: Record<string, string>; dataset_version: number; config: Config; libraries: Record<string, string>; sha256: string }; created_at: string };
export type Prediction = { predictions: (number | string)[]; probabilities: Record<string, number>[] | null; latency_ms: number; performance: Record<string, number> | null };
export type DriftFeature = { feature: string; method: string; statistic: number; p_value: number | null; adjusted_p_value: number | null; flagged: boolean; baseline_missing: number; current_missing: number; baseline: Record<string, number>; current: Record<string, number> };
export type Drift = { model_id: string; rows: number; features: DriftFeature[] };
export type Event = { id: number; action: string; resource_id: string; details: Record<string, unknown>; created_at: string };

export const algorithms: Record<Algorithm, string> = {
  logistic: "Logistic regression", linear: "Linear regression", random_forest: "Random forest",
  gradient_boosting: "Gradient boosting", xgboost: "XGBoost", torch_mlp: "PyTorch MLP",
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    const fallback = `Request failed (HTTP ${response.status}).`;
    const error = await response.json().catch(() => ({}));
    const detail = error.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((issue: { loc?: string[]; msg: string }) =>
        `${issue.loc?.slice(1).join(".") || "Input"}: ${issue.msg}`).join(" ")
      : fallback);
  }
  return response.json() as Promise<T>;
}

export function json(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const active = (experiment: Experiment) => ["queued", "running"].includes(experiment.status);
export const friendly = (key: string) => key.replaceAll("_", " ");
export const number = (value: number) => Intl.NumberFormat("en", { maximumSignificantDigits: 5 }).format(value);
export const stamp = (value: string) => new Date(/(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : value + "Z").toLocaleString();
