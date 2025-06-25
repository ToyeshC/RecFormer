## Dataset Structure Comparison: Side-by-Side

### Training Data Files

| Arts Dataset | MIND Dataset | Purpose |
|-------------|-------------|----------|
| **Arts.train.inter** | **train.json** | Training sequences |
| `user_id:token   item_id_list:token_seq  item_id:token` | `[["U13740", 2690, 0], ["U13740", 3934, 0], ...]` | Format |
| `0       0 1 2 3 4 5 6   7` | Each triplet: [user_id, item_id, label] | Example |
| Simple sequence prediction format | List of user-item interactions with labels | Structure |

### Item Metadata Files

| Arts Dataset | MIND Dataset | Purpose |
|-------------|-------------|----------|
| **Arts.text** | **meta_data.json** | Item content/descriptions |
| `item_id:token   text:token_seq` | `{"N61837": {"text": "The Cost of Trump's..."}}` | Format |
| `8862933177   Moleskine Passion Journal - Film...` | Rich news article content with full text | Content Type |
| Product descriptions | News articles with detailed content | Nature |

### Original Raw Data (MIND only)

| File | Structure | Example |
|------|-----------|---------|
| **behaviors.tsv** | `impression_id  user_id  timestamp  history  impressions` | `1  U13740  11/11/2019 9:05:58 AM  N55189 N42782...  N55689-1 N35729-0` |
| **news.tsv** | `news_id  category  subcategory  title  abstract  url  entities...` | `N61837  news  newsworld  The Cost of Trump's Aid...` |

### Index Mapping Files

| Arts Dataset | MIND Dataset | Purpose |
|-------------|-------------|----------|
| **Arts.item2index** | **smap.json** | Item ID to index mapping |
| **Arts.user2index** | Part of smap.json | User ID to index mapping |
| Separate files for users/items | Combined mapping file | Organization |

### Key Structural Differences

1. **Data Format**: 
   - Arts: Tab-separated values with simple sequences
   - MIND: JSON format with rich metadata and complex structures

2. **Training Examples**:
   - Arts: `user_id -> item_sequence -> next_item` (sequence prediction)
   - MIND: `[user_id, item_id, label]` triplets (interaction prediction)

3. **Item Content**:
   - Arts: Product descriptions (crafts, arts supplies)
   - MIND: News articles with categories, abstracts, entities

4. **Temporal Information**:
   - Arts: Implicit in sequence order
   - MIND: Explicit timestamps in behaviors.tsv

5. **Labels**:
   - Arts: Next item in sequence (implicit positive)
   - MIND: Explicit click labels (0=not clicked, 1=clicked) 