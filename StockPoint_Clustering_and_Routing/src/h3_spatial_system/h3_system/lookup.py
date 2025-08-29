

# class AddressLookup

# ### Basic Address Lookup
# ```python
# from src.h3_system.lookup import AddressLookup

# lookup = AddressLookup()
# address = lookup.get_address_by_h3("8c1234567890abc")
# print(address)
# # Output: {
# #   "h3_id": "8c1234567890abc",
# #   "primary_address_id": "NG-LA-IK-WA-X7M9",
# #   "admin_assignment": {...}
# # }
# ```

# ### Reverse Geocoding
# ```python
# address = lookup.get_address_by_coordinates(6.5244, 3.3792)
# print(address.primary_address_id)
# # Output: "NG-LA-IK-WA-X7M9"
# ```