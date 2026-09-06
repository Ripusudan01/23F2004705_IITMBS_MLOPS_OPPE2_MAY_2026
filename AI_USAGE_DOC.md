# OPPE-2 AI Usage Documentation

## AI Tool Utilized

### ChatGPT

ChatGPT was used as an AI assistance tool during the development and debugging of the OPPE-2 MLOps project.

### Purpose of AI Assistance

ChatGPT was used for:

- Understanding the OPPE-2 problem statement and deliverables.
- Planning the implementation of the MLOps pipeline.
- Understanding and implementing SHAP-based model explainability.
- Understanding and implementing Fairlearn fairness analysis using `age` as the sensitive attribute.
- Debugging Python, FastAPI, Docker, Kubernetes, GKE, and GitHub Actions issues.
- Understanding GKE deployment, Kubernetes HPA, Artifact Registry, and CI/CD.
- Preparing and validating the 100-row prediction and logging workflow.
- Understanding `wrk` stress testing and interpreting throughput, latency, concurrency, and timeout results.
- Implementing and interpreting input drift detection using PSI.
- Preparing documentation and checking the final OPPE-2 deliverables.

All implementation and commands were executed and verified by the student in the provided GCP/GKE environment.

---

## Prompts / Assistance Used

Examples of AI-assisted requests included:

1. Understanding the OPPE-2 problem statement and breaking it into the required deliverables.

2. Getting guidance for implementing SHAP explainability and interpreting feature importance in plain English.

3. Getting guidance for implementing Fairlearn fairness analysis with `age` as the sensitive attribute.

4. Getting step-by-step guidance for Dockerizing the FastAPI heart disease prediction model.

5. Getting guidance for deploying the Dockerized API to GKE using Kubernetes Deployment, Service, and HPA.

6. Getting guidance for generating a 100-row random prediction dataset and sending individual requests to the deployed API.

7. Getting guidance for verifying per-request prediction logs using Kubernetes logs and GCP Cloud Logging.

8. Getting guidance for configuring and running `wrk` with more than 2,000 concurrent connections and interpreting the resulting performance metrics.

9. Getting guidance for implementing input drift detection by comparing the training dataset with the generated 100-row prediction dataset.

10. Getting guidance for configuring GitHub Actions CI/CD to build and push the Docker image to Artifact Registry and deploy it to GKE.

11. Getting debugging assistance for the GitHub Actions GCP authentication failure and correcting the GitHub repository secret containing the service-account credentials.

---

## Shared ChatGPT Conversation

Public shared conversation containing the AI-assisted interaction:

https://chatgpt.com/share/6a6f4508-7fd4-83ee-9520-6006e0b82230

---

## Student Responsibility

AI assistance was used for guidance, debugging, explanation, and implementation support. The resulting commands, code, configurations, and outputs were executed and verified by the student in the OPPE-2 environment.
