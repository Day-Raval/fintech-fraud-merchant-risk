# Kubernetes Manifests

This folder contains Kubernetes manifests for deploying:
- FastAPI service (`fraud-api`) on port `8000`
- Streamlit dashboard (`fraud-dashboard`) on port `8501`

## 1) Build/tag images for your cluster

Example tags used by manifests:
- `fintech-risk-api:latest`
- `fintech-risk-dashboard:latest`

## 2) Apply manifests

```bash
kubectl apply -k k8s/
```

## 3) Verify

```bash
kubectl -n fintech-risk get pods,svc,pvc,ingress
```

## Notes

- Services are `ClusterIP`; expose with ingress (`fintech-risk.local`) or `kubectl port-forward`.
- PVCs use default StorageClass with `ReadWriteOnce`.
- If your cluster cannot pull local images, push images to a registry and update image names in deployment files.
