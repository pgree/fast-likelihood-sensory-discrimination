# Supertests

A supertest (see [Schematic](https://schematic.tech)) is a Python function that states a 
property with `assume(...)` and
`assert ...`. [Schematic's Super](https://docs.schematic.tech/super)
either proves that the assertion holds for every input the
`assume` admits, or returns an input where it fails.

Super proved both supertests in `tail.py`. Together they show that the d = 2
network, `hhat` in `network.py`, satisfies `1 - hhat <= 2.2e-5` for every
`eta >= 0` with `||eta|| >= 12` and every `1e-3 <= lambda_2 <= lambda_1 <= 1.499`,
although it was trained only on `||eta|| <= 12`.

- `network_within_2_2e_5_of_one_beyond_R`: the bound for `12 <= ||eta|| <= 1000`.
- `logit_grows_at_least_linearly`: the logit grows at least like `||eta||`; with
  the constant `K` from `make_network.py` this gives `1 - hhat < 1e-294` beyond 1000.

Reading the claims and evaluating `network.py`, which uses only the standard
library, needs neither Super nor an account. Rerunning the proofs needs
`pip install schematic-supertest`, a Schematic account, and
`super check supertests/tail.py`. Dates and details are in `tail.py`.
