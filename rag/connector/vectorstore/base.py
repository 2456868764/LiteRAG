from abc import ABC, abstractmethod
class VectorStore(ABC):
    """Abstract base class for vector store implementations.

    This class defines a common interface for various vector store implementations,
    allowing for consistent interaction with vector-based data structures across different
    implementations. It includes methods for creating, dropping, and clearing vector stores,
    as well as adding, deleting, updating, and searching documents within the stores.
    """

    @abstractmethod
    def create_vectorstore(self):
        """Creates a new vector store.

        This method is responsible for initializing a new vector store, which can involve
        setting up necessary data structures or connecting to a vector database.
        """

    @abstractmethod
    def drop_vectorstore(self):
        """Drops the current vector store.

        This method is used to remove or delete the entire vector store. It should handle
        the proper disposal of resources and data associated with the vector store.
        """

    @abstractmethod
    def clear_vectorstore(self):
        """Clears the current vector store.

        This method clears the contents of the vector store without deleting the store itself.
        It is useful for resetting the store while maintaining the underlying structure.
        """

    @abstractmethod
    def add_doc(self, file, docs):
        """Adds documents to the vector store.

        Args:
            file (str): The name of the file to which the documents belong.
            docs (list): A list of documents to be added to the vector store.

        This method should process the documents and add them to the specified file within
        the vector store.
        """

    @abstractmethod
    def delete_doc(self, filename):
        """Deletes a document from the vector store.

        Args:
            filename (str): The name of the file to be deleted.

        This method should remove the specified document from the vector store.
        """

    @abstractmethod
    def update_doc(self, file, docs):
        """Updates documents in the vector store.

        Args:
            file (str): The name of the file containing the documents to be updated.
            docs (list): A list of updated documents.

        This method should update the specified documents within the vector store.
        """

    @abstractmethod
    def search_docs(self, text, top_k, threshold, **kwargs):
        """Searches for documents containing the specified text.

        Args:
            text (str): The text to search for.
            top_k (int): The number of top results to return.
            threshold (float): The minimum relevance score for a document to be included in the results.
            **kwargs: Additional keyword arguments for customizing the search.

        Returns:
            A list of documents that match the search criteria.
        """

    @abstractmethod
    def search_docs_by_vector(self, embedding, top_k, threshold, **kwargs):
        """Searches for documents similar to the specified vector.

        Args:
            embedding (list): The vector embedding to compare against.
            top_k (int): The number of top results to return.
            threshold (float): The minimum similarity score for a document to be included in the results.
            **kwargs: Additional keyword arguments for customizing the search.

        Returns:
            A list of documents that are similar to the specified vector.
        """

    @abstractmethod
    def search_docs_by_mmr(self, text, top_k, fetch_k, lambda_mult, **kwargs):
        """Searches for documents using Maximal Marginal Relevance (MMR).

        Args:
            text (str): The text to search for.
            top_k (int): The number of top results to return.
            fetch_k (int): The number of top candidates to fetch before applying MMR.
            lambda_mult (float): The lambda parameter for balancing relevance and diversity in MMR.
            **kwargs: Additional keyword arguments for customizing the search.

        Returns:
            A list of documents selected based on MMR criteria.
        """

