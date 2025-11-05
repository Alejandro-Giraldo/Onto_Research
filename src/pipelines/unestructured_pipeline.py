from prefect import flow, task

@task
def process_unstructured_data():
    """
    Process unstructured data
    """
    return "Processed"

@task
def enrich_data():
    """
    Enrich data
    """
    return "Enriched"

@task
def store_data():
    """
    Store data in neo4j
    """
    return "Stored"


@flow
def process_unstructured_data_flow():
    """
    Process unstructured data flow
    """
    process_unstructured_data(docs)
    enrich_data(data)
    store_data(data)
    return "Done"


if __name__ == "__main__":
    process_unstructured_data_flow.serve(
        name="unstructured-data-flow",
        #cron="0 0 * * *",
    )