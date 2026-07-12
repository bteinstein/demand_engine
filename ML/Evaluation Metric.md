## 📊 Evaluation Metrics Glossary
 

- **`Top_K`**  
  The number of top recommendations (e.g., top 5 SKUs) evaluated for each customer. It shows how many items we’re suggesting to prioritize.

- **`Top_Predicted_SKUs`**  
  The list of SKUs recommended for a customer, ranked by their predicted purchase likelihood. These are the items we expect the customer to buy.

- **`Historical_Purchases`**  
  The SKUs a customer has previously purchased. This is the customer’s actual buying history for comparison.

- **`Historical_Categories`**  
  The product categories (e.g., Electronics, Clothing) a customer has purchased from. It shows their category preferences.

### 📈 Accuracy & Performance Metrics

- **`SKU_Precision`**  
  The percentage of recommended SKUs that the customer has actually purchased before. Higher precision means our recommendations are accurate.

- **`SKU_Recall`**  
  The percentage of a customer’s historical purchases that appear in our recommendations. Higher recall means we’re capturing more of their past purchases.

- **`SKU_F1_Score`**  
  A balanced measure combining precision and recall. It indicates overall recommendation quality, useful when precision and recall need to be weighed together.

### 🧠 Relevance & Diversity

- **`Category_Overlap_Proportion`**  
  The percentage of recommended SKU categories that match the categories the customer has purchased from. It shows if we’re recommending the right types of products.

- **`Mean_Purchase_Frequency`**  
  The average number of times the recommended SKUs were purchased by the customer in the past. Higher values suggest we’re recommending frequently bought items.

- **`NDCG (Normalized Discounted Cumulative Gain)`**  
  A score (0 to 1) measuring how well we rank relevant SKUs, with higher scores for relevant items appearing at the top of the list. It evaluates the quality of our recommendation order.

- **`MRR (Mean Reciprocal Rank)`**  
  A score (0 to 1) based on the position of the first relevant SKU in the recommendation list. Higher scores mean relevant items are ranked earlier.

### ⏱️ Temporal Insights

- **`Mean_Recency_Weeks`**  
  The average time (in weeks) since the customer last purchased the recommended SKUs. Lower values indicate we’re recommending recently bought items.

- **`Recent_Matches`**  
  The number of recommended SKUs purchased by the customer in the last 2 weeks. It shows if our recommendations align with recent buying behavior.

### 🌐 Category Diversity

- **`Unique_Categories`**  
  The number of different product categories in the recommended SKUs. More categories suggest diverse recommendations.

- **`Category_Entropy`**  
  A measure of how varied the categories are in the recommendations. Higher values indicate more diverse category recommendations, avoiding overly narrow suggestions.

### 📝 Recommendation Context

- **`Validation_Notes`**  
  Detailed explanations for each recommended SKU, noting whether it was purchased before, its category purchase history, or if it’s entirely new. It provides context for why recommendations were made.

- **`Recent_Purchases`**  
  The customer’s most recent purchases (SKUs, categories, and weeks). It helps compare recommendations against what the customer bought lately.