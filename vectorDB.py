import numpy as np

class VectorDB:
    def __init__(self):
        self.embeddings = []
        self.metadata = {}

    def search(self, query, top_k = 5):
        all_sims = []
        for i, emb in enumerate(self.embeddings):
            all_sims.append((i,self.cosine_similarity(query, emb)))

        all_sims = sorted(all_sims, key=lambda x: x[1], reverse=True)

        top_k_indices = [x[0] for x in all_sims[:top_k]]

        all_meta = [self.metadata[i] for i in top_k_indices]

        return all_meta

    def insert(self, embedding, meta):
        self.embeddings.append(embedding)
        self.metadata[len(self.embeddings)-1] = meta


    def cosine_similarity(self,v1, v2):
        dot_product = np.dot(v1, v2)

        norm_matrix1 = np.linalg.norm(v1) 
        norm_matrix2 = np.linalg.norm(v2)  
        similarity = dot_product / (norm_matrix1 * norm_matrix2 + 1e-10)

        return similarity


if __name__ == '__main__':
    vb = VectorDB()
    vb.insert(np.array([1,0,1]), meta='hello')
    vb.insert(np.array([0,0,1]), meta='hi')

    print(vb.search(np.array([1,0,1]), top_k = 1))