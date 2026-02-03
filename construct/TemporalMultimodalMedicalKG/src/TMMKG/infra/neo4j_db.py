from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


def create_neo4j_driver(
    uri: str,
    user: str,
    password: str,
    verify: bool = True,
):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    if verify:
        try:
            with driver.session() as session:
                session.run("RETURN 1").consume()
        except ServiceUnavailable as e:
            driver.close()
            raise RuntimeError("Failed to connect to Neo4j") from e

    return driver
