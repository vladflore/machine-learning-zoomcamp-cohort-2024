```shell
kubectl get hpa subscription-hpa --watch

kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

kubectl get hpa

kubectl autoscale deployment subscription --name subscription-hpa --cpu-percent=20 --min=1 --max=3

kubectl port-forward service/subscription 9696:80

kind load docker-image zoomcamp-model:3.11.5-hw10

kubectl cluster-info

kind create cluster
```

Install [k9s](https://k9scli.io/topics/install/)

```shell
brew install derailed/k9s/k9s
```
