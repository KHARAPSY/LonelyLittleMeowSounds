PROD_COMPOSE=dockers/docker-compose.prod.yml
DEV_COMPOSE=dockers/docker-compose.dev.yml
TEST_COMPOSE=dockers/docker-compose.test.yml

.PHONY: prod
prod:
	docker compose -f $(PROD_COMPOSE) up --build

.PHONY: prod-restart
prod-restart:
	docker compose -f $(PROD_COMPOSE) restart

.PHONY: dev
dev:
	docker compose -f $(DEV_COMPOSE) up --build

.PHONY: dev-restart
dev-restart:
	docker compose -f $(DEV_COMPOSE) restart

.PHONY: test
test:
	docker compose -f $(TEST_COMPOSE) up --build

.PHONY: logs
logs:
	docker compose logs -f

.PHONY: clean
clean:
	docker compose down -v --rmi all --remove-orphans

.PHONY: clean-logs
clean-logs:
	rm logs/*