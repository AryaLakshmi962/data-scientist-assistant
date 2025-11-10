import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

const ReportView = ({ report }) => {
  if (!report) {
    return (
      <div className="text-center text-gray-500 mt-10">
        <p>No report data available yet. Upload a dataset first!</p>
      </div>
    );
  }

  const {
    cleaned_path,
    column_types,
    columns,
    datetime_columns,
    detected_target,
    dropped_columns,
    missing_counts,
    model_metrics,
    model_path,
    rows
  } = report;

  const featureData =
    model_metrics?.feature_importances?.map((item) => ({
      name: item.feature,
      importance: item.importance,
    })) || [];

  return (
    <div className="p-8 space-y-8">
      {/* Summary */}
      <Card className="shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-semibold mb-4">📊 Dataset Summary</h2>
          <p><strong>Total Rows:</strong> {rows}</p>
          <p><strong>Total Columns:</strong> {columns}</p>
          <p><strong>Detected Target:</strong> {detected_target || "None"}</p>
          <p><strong>Cleaned File:</strong> {cleaned_path}</p>
          <p><strong>Model Path:</strong> {model_path}</p>
        </CardContent>
      </Card>

      {/* Column Info */}
      <Card className="shadow-md">
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-4">🧩 Column Types</h2>
          <table className="w-full border border-gray-300">
            <thead>
              <tr className="bg-gray-100">
                <th className="border p-2 text-left">Column</th>
                <th className="border p-2 text-left">Type</th>
                <th className="border p-2 text-left">Missing Values</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(column_types).map((col) => (
                <tr key={col}>
                  <td className="border p-2">{col}</td>
                  <td className="border p-2">{column_types[col]}</td>
                  <td className="border p-2">{missing_counts[col]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Dropped Columns */}
      <Card className="shadow-md">
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-4">🗑️ Dropped Columns</h2>
          {dropped_columns.length > 0 ? (
            <ul className="list-disc pl-6">
              {dropped_columns.map((col) => (
                <li key={col}>{col}</li>
              ))}
            </ul>
          ) : (
            <p>No columns dropped ✅</p>
          )}
        </CardContent>
      </Card>

      {/* Feature Importance Chart */}
      {featureData.length > 0 && (
        <Card className="shadow-md">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-4">🔥 Top Features by Importance</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="importance" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ReportView;
