"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function Panel({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return <section className="panel"><header><h2>{title}</h2>{hint && <p>{hint}</p>}</header>{children}</section>;
}

export function Notice({ error }: { error: unknown }) {
  return error ? <p className="notice" role="alert">{error instanceof Error ? error.message : String(error)}</p> : null;
}

export function Grid({ headers, rows }: { headers: string[]; rows: React.ReactNode[][] }) {
  return <div className="table-wrap"><table><thead><tr>{headers.map((item, index) =>
    <th key={index} scope="col">{item}</th>)}</tr></thead><tbody>{rows.map((row, index) =>
      <tr key={index}>{row.map((item, cell) => <td key={cell}>{item}</td>)}</tr>)}</tbody></table></div>;
}

export function Chart({ data, label, fields, description }: {
  data: Record<string, string | number>[]; label: string; fields: string[]; description: string;
}) {
  return <div className="chart" role="img" aria-label={description}>
    <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 420, height: 245 }}>
      <BarChart data={data} margin={{ top: 8, bottom: 12, right: 15 }}>
        <CartesianGrid stroke="#263347" vertical={false} />
        <XAxis dataKey={label} tick={{ fill: "#9aa9c1", fontSize: 11 }} />
        <YAxis tick={{ fill: "#9aa9c1", fontSize: 11 }} />
        <Tooltip contentStyle={{ backgroundColor: "#141e2f", border: "1px solid #324158" }} />
        {fields.map((field, i) => <Bar key={field} dataKey={field} fill={i ? "#92a9fb" : "#7de4c1"} radius={[4, 4, 0, 0]} />)}
      </BarChart>
    </ResponsiveContainer>
  </div>;
}
