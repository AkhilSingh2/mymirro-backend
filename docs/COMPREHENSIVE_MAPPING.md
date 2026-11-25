# Comprehensive Product Type Mapping & Compatibility Rules

## Current Product Types in Database

### Topwear Products:
- **T-Shirt** / **tshirt**
- **Shirt** / **shirt** 
- **Top** / **top**
- **Sweater**
- **Hoodie**
- **Blazer**
- **Jacket**
- **Coat**
- **Kurta**
- **blouse**

### Bottomwear Products:
- **Shorts**
- **Pants** / **pants**
- **Jeans**
- **Joggers**
- **Trousers** / **trousers**
- **Leggings**
- **cargos**

### Full Body Products:
- **Dress** / **dress**
- **maxi dress**
- **athleisure**

---

## Current Compatibility Rules (Phase 2)

### Bottomwear Compatibility Groups:
```python
compatible_groups = {
    # Shorts group
    'shorts': ['shorts', 'denim_shorts', 'cargo_shorts', 'athletic_shorts'],
    
    # Pants group  
    'pants': ['pants', 'casual_pants', 'formal_pants', 'dress_pants', 'trousers'],
    
    # Jeans group
    'jeans': ['jeans', 'denim_pants', 'skinny_jeans', 'straight_jeans'],
    
    # Chinos group
    'chinos': ['chinos', 'khaki_pants', 'casual_pants'],
    
    # Joggers group
    'joggers': ['joggers', 'track_pants', 'sweatpants'],
    
    # Cargos group
    'cargos': ['cargos', 'cargo_pants', 'cargo_shorts'],
    
    # Leggings group
    'leggings': ['leggings', 'athletic_leggings', 'yoga_pants']
}
```

---

## Suggested Expansions for Larger Product Pool

### 1. **Smart Product Type Grouping**

#### For Shorts Main Product:
```python
shorts_compatible = [
    'shorts',           # Exact match
    'cargos',           # Cargo shorts
    'denim_shorts',     # Denim shorts
    'athletic_shorts',  # Athletic shorts
    'cargo_shorts'      # Cargo shorts variant
]
```

#### For Joggers Main Product:
```python
joggers_compatible = [
    'joggers',          # Exact match
    'track_pants',      # Similar athletic style
    'sweatpants',       # Similar comfort style
    'cargos',           # Cargo pants (casual)
    'pants'             # General pants (if casual)
]
```

#### For Jeans Main Product:
```python
jeans_compatible = [
    'jeans',            # Exact match
    'denim_pants',      # Denim variants
    'skinny_jeans',     # Jeans styles
    'straight_jeans',   # Jeans styles
    'pants'             # General pants (if similar style)
]
```

#### For Pants Main Product:
```python
pants_compatible = [
    'pants',            # Exact match
    'trousers',         # Formal pants
    'casual_pants',     # Casual variants
    'chinos',           # Chino pants
    'cargos',           # Cargo pants
    'joggers'           # Casual joggers
]
```

### 2. **Style-Based Compatibility**

#### Casual Style Group:
```python
casual_bottoms = ['shorts', 'joggers', 'cargos', 'casual_pants', 'jeans']
casual_tops = ['t-shirt', 'tshirt', 'hoodie', 'sweater', 'shirt']
```

#### Formal Style Group:
```python
formal_bottoms = ['pants', 'trousers', 'dress_pants', 'chinos']
formal_tops = ['shirt', 'blazer', 'coat', 'jacket']
```

#### Athletic Style Group:
```python
athletic_bottoms = ['shorts', 'joggers', 'leggings', 'track_pants']
athletic_tops = ['t-shirt', 'tshirt', 'hoodie', 'athleisure']
```

### 3. **Occasion-Based Compatibility**

#### Work/Office:
```python
work_bottoms = ['pants', 'trousers', 'chinos', 'dress_pants']
work_tops = ['shirt', 'blazer', 'coat', 'sweater']
```

#### Casual/Everyday:
```python
casual_bottoms = ['jeans', 'shorts', 'joggers', 'cargos', 'pants']
casual_tops = ['t-shirt', 'tshirt', 'hoodie', 'shirt', 'sweater']
```

#### Athletic/Sport:
```python
athletic_bottoms = ['shorts', 'joggers', 'leggings', 'track_pants']
athletic_tops = ['t-shirt', 'tshirt', 'hoodie', 'athleisure']
```

---

## Recommended Implementation Strategy

### Phase 1: Immediate Expansion
1. **Add cargos to shorts compatibility** (as you suggested)
2. **Add joggers to pants compatibility**
3. **Add jeans to pants compatibility** (for casual styles)

### Phase 2: Style-Based Expansion
1. **Implement style detection** based on product attributes
2. **Use style groups** for broader compatibility
3. **Maintain quality** with style-specific scoring

### Phase 3: Smart Fallback
1. **Best Available strategy** (already implemented)
2. **Gradual relaxation** of constraints
3. **Quality preservation** with minimum thresholds

---

## Current Configuration (After Changes)

```python
similarity_config = {
    'semantic_weight': 4.0,           # Core AI matching
    'style_harmony_weight': 3.5,      # Advanced style compatibility  
    'color_harmony_weight': 3.0,      # Sophisticated color theory
    'pattern_compatibility_weight': 2.0,  # Pattern mixing intelligence
    'occasion_weight': 2.2,           # Occasion-specific matching
    'diversity_bonus': 0.8,           # Encourage variety in results
    'confidence_threshold': 0.1,      # Lowered for more outfits
    'min_similar_outfits': 5,         # Minimum outfits to return
    'max_similar_outfits': 20,        # Maximum outfits to return
    'candidate_pool_size': 500,       # Increased for more candidates
    'fallback_strategy': 'best_available'  # Best Available fallback
}
```

---

## Next Steps for Testing

1. **Test current changes** with outfit `main_68_1751205758_1`
2. **Implement cargos → shorts compatibility**
3. **Monitor quality** of recommendations
4. **Adjust thresholds** based on results
5. **Expand compatibility groups** gradually

Would you like me to implement any of these specific compatibility expansions? 