import { expect, test } from "@playwright/test";
import path from "node:path";

test("uploads a CSV, trains asynchronously, registers a model, predicts and checks drift", async ({ page }) => {
  const runName = `Iris browser ${Date.now()}`;
  const csv = path.resolve("../samples/iris.csv");
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Workspace" });
  await nav.getByRole("button", { name: "Datasets" }).click();
  await page.getByLabel("Dataset name").fill(runName);
  await page.getByLabel("CSV file").setInputFiles(csv);
  await page.getByRole("button", { name: "Upload CSV" }).click();
  await expect(page.getByRole("heading", { name: "Data quality" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "petal_length", exact: true }).first()).toBeVisible();

  await nav.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel("Experiment name").fill(runName);
  await page.getByLabel("Dataset version").selectOption({ label: `${runName} · v1` });
  await expect(page.getByLabel("Target column")).toHaveValue("species");
  await page.getByRole("button", { name: "Configure preprocessing" }).click();
  await page.getByLabel("Optuna trials").fill("2");
  await page.getByRole("button", { name: "Create experiment" }).click();
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page.getByRole("heading", { name: "Held-out evaluation" })).toBeVisible({ timeout: 45_000 });
  await expect(page.getByRole("heading", { name: "Confusion matrix" })).toBeVisible();
  await expect(page.getByText("MLflow run")).toBeVisible({ timeout: 15_000 });
  await nav.getByRole("button", { name: "Compare" }).click();
  await expect(page.getByRole("cell", { name: runName })).toBeVisible();

  await nav.getByRole("button", { name: "Models" }).click();
  await page.getByLabel("Model version").selectOption({ label: `${runName} · v1` });
  await page.getByLabel("Batch CSV").setInputFiles(csv);
  await page.getByRole("button", { name: "Run prediction" }).click();
  await expect(page.getByRole("heading", { name: "Prediction results" })).toBeVisible();
  await expect(page.getByText(/150 rows ·/)).toBeVisible();
  await expect(page.getByRole("cell", { name: "setosa", exact: true }).first()).toBeVisible();
  await page.getByLabel("Operation").selectOption("drift");
  await page.getByLabel("Batch CSV").setInputFiles(path.resolve("../samples/iris_shifted.csv"));
  await page.getByRole("button", { name: "Analyze drift" }).click();
  const drift = page.getByRole("row").filter({ hasText: "sepal_length" });
  await expect(drift.getByRole("cell", { name: "Review" })).toBeVisible();
  await nav.getByRole("button", { name: "Activity" }).click();
  await expect(page.getByRole("cell", { name: "drift measured" }).first()).toBeVisible();
});
