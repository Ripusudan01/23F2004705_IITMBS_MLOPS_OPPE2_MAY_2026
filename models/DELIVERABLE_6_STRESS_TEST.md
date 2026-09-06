# Deliverable 6 — Stress Testing with wrk

## Objective

Evaluate the deployed Heart Disease Prediction API on GKE under high concurrency using `wrk` with more than 2000 concurrent connections.

## Test Configuration

- API endpoint: `http://34.93.99.223/predict`
- Tool: `wrk`
- Threads: 8
- Concurrent connections: 2001
- Duration: 60 seconds
- Timeout: 10 seconds
- Request type: HTTP POST
- Kubernetes HPA minimum replicas: 1
- Kubernetes HPA maximum replicas: 3
- HPA CPU target: 60%

## Results

- Total requests: 9,937
- Throughput: 165.44 requests/second
- Average latency: 6.74 seconds
- P50 latency: 6.91 seconds
- P75 latency: 8.05 seconds
- P90 latency: 8.96 seconds
- P99 latency: 9.86 seconds
- Maximum latency: 10.00 seconds
- Connection errors: 0
- Read errors: 697
- Write errors: 30
- Timeouts: 948

## Kubernetes Autoscaling

During the stress test, the HPA reached its configured maximum of 3 replicas.

Observed during the test:

- CPU utilization: 397% / 60% target
- Minimum replicas: 1
- Maximum replicas: 3
- Replicas: 3

The HPA successfully responded to the increased CPU utilization and scaled the API deployment to its configured maximum of three pods.

## Pod Status

After the stress test, all three API pods were 1/1 Running.

Each pod recorded one restart during the stress-test period. This should be investigated further for a production deployment.

## Performance Analysis

The API achieved 165.44 requests/second under 2001 concurrent connections.

Latency increased substantially under this workload. The P99 latency reached 9.86 seconds, which was close to the configured 10-second client timeout.

The test reported 948 timeouts, 697 read errors, and 30 write errors, while connection errors remained at zero.

This indicates that the deployment could accept connections but became heavily saturated at 2001 concurrent connections.

## Autoscaling Analysis

The HPA successfully detected the high CPU workload and scaled the deployment to 3 replicas.

Because the HPA maximum was configured as 3 replicas, no additional API pods could be created.

After the stress test, CPU utilization returned to approximately 3%.

## Conclusion

The GKE deployment was successfully stress-tested using 2001 concurrent connections for 60 seconds.

The deployment achieved 165.44 requests/second and the HPA scaled the application to its configured maximum of 3 replicas.

However, the extreme concurrency caused significant performance degradation, with P99 latency of 9.86 seconds and 948 request timeouts. There were also 697 read errors and 30 write errors.

The results demonstrate horizontal scaling within the configured 1–3 replica range, but the current configuration does not provide comfortable capacity for 2001 concurrent connections.

Potential production improvements include increasing pod CPU capacity, tuning application/server concurrency, optimizing the API workload, and increasing the HPA maximum replica count if higher concurrency is required.

All values in this report are based on the actual wrk and Kubernetes observations from the stress test.
