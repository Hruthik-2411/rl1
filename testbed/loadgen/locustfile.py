from locust import HttpUser, between, task


class Shopper(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(5)
    def browse(self):
        self.client.get("/items")

    @task(1)
    def health(self):
        self.client.get("/health")
