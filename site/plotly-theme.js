document.addEventListener("DOMContentLoaded", function () {
  const theme = localStorage.getItem("quarto-color-scheme") || "light";
  const plots = document.querySelectorAll(".plotly-graph-div");

  plots.forEach(plot => {
    Plotly.relayout(plot, {
      paper_bgcolor: theme === "dark" ? "#1e1e2e" : "#ffffff",
      plot_bgcolor: theme === "dark" ? "#1e1e2e" : "#ffffff",
      font: {
        color: theme === "dark" ? "#cdd6f4" : "#1e1e2e"
      }
    });
  });
});
