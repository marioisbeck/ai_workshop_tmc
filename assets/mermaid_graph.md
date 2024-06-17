# Mermaid Graph
"""
graph TD
    subgraph Ingestion
        direction TB
        Load(Load):::mainPoint
        Load:::description --> Split(Split):::mainPoint
        Split:::description --> Embed(Embed):::mainPoint
        Embed:::description --> Store(Store):::mainPoint
        Store:::description
    end

    Similarity_Search(Similarity Search):::mainPoint
    Similarity_Search:::description --> Combine_Context(Combine Context):::mainPoint
    Combine_Context:::description --> Response_Generation(Response Generation):::mainPoint
    Response_Generation:::description

    Store --> Similarity_Search

    classDef mainPoint fill:#f9f,stroke:#333,stroke-width:4px;
    classDef description fill:#bbf,stroke:#333,stroke-width:2px;
    class Load fill:#bbf,stroke:#333,stroke-width:2px;
    class Split fill:#bbf,stroke:#333,stroke-width:2px;
    class Embed fill:#bbf,stroke:#333,stroke-width:2px;
    class Store fill:#bbf,stroke:#333,stroke-width:2px;
    class Similarity_Search fill:#bbf,stroke:#333,stroke-width:2px;
    class Combine_Context fill:#bbf,stroke:#333,stroke-width:2px;
    class Response_Generation fill:#bbf,stroke:#333,stroke-width:2px;
"""