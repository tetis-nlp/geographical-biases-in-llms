import pandas as pd

# from https://arxiv.org/pdf/2104.03090
data = {
    "Data": [
        "2016 Ecuador Earthquake", "2016 Canada Wildfires", "2016 Italy Earthquake", 
        "2016 Kaikoura Earthquake", "2016 Hurricane Matthew", "2017 Sri Lanka Floods", 
        "2017 Hurricane Harvey", "2017 Hurricane Irma", "2017 Hurricane Maria", 
        "2017 Mexico Earthquake", "2018 Maryland Floods", "2018 Greece Wildfires", 
        "2018 Kerala Floods", "2018 Hurricane Florence", "2018 California Wildfires", 
        "2019 Cyclone Idai", "2019 Midwestern U.S. Floods", "2019 Hurricane Dorian", 
        "2019 Pakistan Earthquake", "Earthquake", "Fire", "Flood", "Hurricane", "All", "Average"
    ],
    "# Cls": [
        8, 8, 6, 9, 9, 8, 9, 9, 9, 8, 8, 9, 9, 9, 10, 10, 7, 9, 8, 9, 10, 10, 10, 10, None
    ],
    "RF": [
        0.784, 0.726, 0.799, 0.660, 0.742, 0.613, 0.719, 0.693, 0.682, 0.800, 0.554, 0.678, 
        0.670, 0.731, 0.676, 0.680, 0.643, 0.688, 0.753, 0.766, 0.685, 0.653, 0.702, 0.707, 0.700
    ],
    "SVM": [
        0.738, 0.738, 0.822, 0.693, 0.700, 0.611, 0.713, 0.695, 0.682, 0.789, 0.620, 0.694, 
        0.694, 0.717, 0.696, 0.730, 0.632, 0.663, 0.766, 0.783, 0.717, 0.693, 0.716, 0.731, 0.710
    ],
    "FastText": [
        0.752, 0.726, 0.821, 0.658, 0.704, 0.575, 0.718, 0.694, 0.688, 0.797, 0.621, 0.667, 
        0.714, 0.735, 0.713, 0.707, 0.624, 0.693, 0.787, 0.789, 0.727, 0.704, 0.730, 0.744, 0.712
    ],
    "BERT": [
        0.861, 0.792, 0.871, 0.768, 0.786, 0.703, 0.759, 0.722, 0.715, 0.845, 0.697, 0.788, 
        0.732, 0.768, 0.760, 0.790, 0.702, 0.691, 0.820, 0.833, 0.771, 0.749, 0.740, 0.758, 0.768
    ],
    "D-BERT": [
        0.872, 0.781, 0.878, 0.743, 0.780, 0.763, 0.743, 0.723, 0.722, 0.854, 0.734, 0.739, 
        0.732, 0.773, 0.767, 0.779, 0.706, 0.691, 0.822, 0.839, 0.771, 0.734, 0.742, 0.758, 0.769
    ],
    "RoBERTa": [
        0.872, 0.791, 0.885, 0.765, 0.815, 0.727, 0.763, 0.730, 0.727, 0.863, 0.760, 0.783, 
        0.745, 0.780, 0.764, 0.796, 0.764, 0.686, 0.834, 0.836, 0.787, 0.758, 0.741, 0.760, 0.781
    ],
    "XLM-R": [
        0.866, 0.768, 0.877, 0.760, 0.784, 0.798, 0.761, 0.717, 0.723, 0.847, 0.798, 0.783, 
        0.746, 0.765, 0.757, 0.793, 0.726, 0.691, 0.827, 0.837, 0.779, 0.755, 0.739, 0.758, 0.777
    ],
    "Event Type": [
        "Earthquake", "Wildfire", "Earthquake", "Earthquake", "Hurricane", "Flood", 
        "Hurricane", "Hurricane", "Hurricane", "Earthquake", "Flood", "Wildfire", 
        "Flood", "Hurricane", "Wildfire", "Hurricane", "Flood", "Hurricane", 
        "Earthquake", "Earthquake", "Wildfire", "Flood", "Hurricane", "All", "Average"
    ],
    'Caution and advice': [30, 106, 10, 493, 36, 28, 541, 613, 220, 35, 70, 26, 139, 1310, 139, 89, 79, 1369, 71, 0, 0, 0, 0, 0, 0 ],
    'Displaced people and evacuations': [3, 380, 3, 87, 38, 9, 688, 755, 131, 4, 3, 7, 56, 637, 57, 8, 802, 0, 0, 0, 0, 0, 0, 0, 0 ],
    'Infrastructure and utility damage': [70, 251, 54, 312, 178, 17, 1217, 1881, 1427, 167, 79, 38, 296, 320, 354, 140, 815, 125, 0, 0, 0, 0, 0, 0, 0 ],
    'Injured or dead people': [555, 4, 174, 105, 224, 46, 698, 894, 302, 254, 56, 495, 363, 297, 433, 14, 60, 401, 152, 0, 0, 0, 0, 0, 0 ],
    'Missing or found people': [10, 0, 7, 3, 0, 4, 10, 8, 11, 14, 140, 20, 7, 0, 19, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ],
    'Not humanitarian': [23, 79, 9, 224, 76, 20, 410, 615, 270, 38, 77, 74, 456, 1060, 80, 389, 874, 213, 0, 0, 0, 0, 0, 0, 0 ],
    'Don’t know or can’t judge': [18, 13, 10, 19, 5, 2, 42, 60, 39, 3, 1, 4, 65, 95, 11, 27, 46, 32, 0, 0, 0, 0, 0, 0, 0 ],
    'Other relevant information': [81, 311, 52, 311, 328, 56, 1767, 2358, 1568, 109, 137, 159, 955, 636, 407, 273, 1444, 154, 0, 0, 0, 0, 0, 0, 0 ],
    'Requests or urgent needs': [91, 20, 30, 24, 53, 34, 333, 126, 711, 61, 1, 25, 590, 54, 143, 46, 179, 19, 0, 0, 0, 0, 0, 0, 0 ],
    'Rescue volunteering or donation effort': [394, 934, 312, 207, 326, 319, 2823, 1590, 1977, 984, 73, 356, 4294, 1478, 1869, 788, 987, 823, 0, 0, 0, 0, 0, 0, 0 ],
    'Sympathy and support': [319, 161, 579, 432, 395, 40, 635, 567, 672, 367, 110, 322, 835, 472, 482, 165, 1083, 1991, 0, 0, 0, 0, 0, 0, 0 ],
    'Total': [1594, 2259, 1240, 2217, 1659, 575, 9164, 9467, 7328, 2036, 747, 1526, 8056, 6359, 7444, 3944, 1930, 7660, 1991, 0, 0, 0, 0, 0, 0 ]
}

df = pd.DataFrame(data)
df["Mean BERT Models"] = df[["BERT", "D-BERT", "RoBERTa", "XLM-R"]].mean(axis=1)
df_ranked = df.sort_values(by="Mean BERT Models", ascending=False)

df_ranked = df_ranked[df_ranked["Data"] != "Earthquake"]
df_ranked = df_ranked[df_ranked["Data"] != "Hurricane"]
df_ranked = df_ranked[df_ranked["Data"] != "Flood"]
df_ranked = df_ranked[df_ranked["Data"] != "Fire"]
df_ranked = df_ranked[df_ranked["Data"] != "Average"]
df_ranked = df_ranked[df_ranked["Data"] != "All"]

print(df_ranked[["Data", "BERT", "D-BERT", "RoBERTa", "XLM-R", "Mean BERT Models"]])

for event in df_ranked["Event Type"].unique():
    print(f"\nEvent: {event}")
    print(df_ranked[df_ranked["Event Type"] == event])

# https://pdf.sciencedirectassets.com/271647/1-s2.0-S0306457322X00083/1-s2.0-S0306457323000778/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAQaCXVzLWVhc3QtMSJHMEUCIQCBf92YPY%2Bs%2BMuAVZu%2Be6sA8By4axz03BQiPBNfZ8%2BWnQIgTPLHgwDrJxJhJMmeeIL9LDqAEs82Nqu3PCUYQboXX2oqvAUIjf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDIjkg66XeYeuPbBepSqQBX6SYzTc7yX6engv9bEyGZ8OHJvJ76lUstb1ddu7NnRWgkr86jx%2FPGtbDITwffcP0rF%2FWtx51R6bVMA8JMo0Y6ROBhciRCN5i7CQlgm4eP8In4U8buvEDLYVxBWRqMMXSiegPvM6q9pVPTSmOKiXjU5Y5TaiR%2BlmLDWutHVSpg9oV0oiMv5g%2BnmDz5WuX9mSNZdtE4uYKFoPwfG848792d3anu3gtwrwnLjCXHJjOk9SgDWbg6swJlC4covzBoua%2FVo8ZX5AC2%2FH4Bq4Tu%2FqvqJeUSIJ3DAxpJ4pox9JDItYYx5lVO6Dp9It4h2lQKkq09eRZJkJu8fVdZGRrQgSZpB9ufdISNzC5Ief42O%2BcBPegfD7qIReMpN90B1N3E21pAuu8paqMYANpaoSIB0z3TNjfRFlj4Iyz71AKjcessVyJXQGlGmiaVe7XIKi0N%2F5qbn4I1rUqFMnrSrMugJqoKY6jf55O8Uy8givr9ylnD032AdmTMYY1ZEqTLfrWglfHDue25%2BAmY6%2F7BfCOCTPu7PeWv4Zh0yfulOZEx41nd1MNWOa1BgkucsfuhcZIcqgFx0OwSHeUpJymMe5VgsgdD7D6WAhnDtJ7iPqut1ixdZpXSbpYrrRuhfcp1QKoNjoiqCx96kVJmVAIJ%2BAt7sOfomg2I1tAFrsG3rQgvCVDqRt8BsYE41NoGgp25BRDWQKC4skFK7wg5Zk1dDwH30rHJ9kYR6U8ayugUJF0HG8EcQWFjFRpkLb5pUXGYKuFP2PFB6ff9ggByFYQlBEwSXV6k4oDVIJdiSXFeyxnQPMoaH%2B6dcbz5qeZOcbL8LVU8wU1i7yBuaoF2BYEhX3yL2oU0qT786I7gzO2Luv%2B%2Fp%2F%2BwXUMLHS9rIGOrEBTo2TIPN9z1ADOdf7Ly9d3gCfhyolzKjs05sCJnxiyxPHPixwctgIeDd%2BiHuBz8F7luMt1LZ1K5HuqAB4XxV%2B%2FCdEVaBujmD%2BhYllBNDHHq%2BSQb3VPjBdT9ZLIHKIX6oxVQ%2BFw3G7xikIUvDYnYCk9XduNPmQs6Zax75wTUSDBZTX5%2FKGWTSTlLCno%2FvHRBl9auWTCu%2B%2FhvP4Oc9kSLrVo6%2Fxk941li0cNZLgqAIomibS&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240603T123451Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYY6UDIRNH%2F20240603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6f05646026d72a1d4ac05d3e808d022ee8cf586ea15f7d4292fb4f71981ad9d4&hash=b2d7e4d16cd65093a411d3aee98b62ad1d09feb46c9014c52dc28555b4b274d3&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0306457323000778&tid=spdf-6b92730c-0290-41f0-a239-a198f817f906&sid=3592cbb8878276457a295398319ec91121c8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=001c585a030105565605&rr=88dfb83bdb495d7c&cc=fr
    

idrisi = {
    'Event': ['Ecuador Earthquake', 'Canada Wildfires', 'Italy Earthquake', 'Kaikoura Earthquake', 'Hurricane Matthew',
              'Sri Lanka Floods', 'Hurricane Harvey', 'Hurricane Irma', 'Hurricane Maria', 'Mexico Earthquake',
              'Maryland Floods', 'Greece Wildfires', 'Kerala Floods', 'Hurricane Florence', 'California Wildfires',
              'Cyclone Idai', 'Midwestern U.S. Floods', 'Hurricane Dorian', 'Pakistan Earthquake', 'Average'],
    'CRF': [0.866, 0.732, 0.558, 0.878, 0.89, 0.856, 0.81, 0.773, 0.864, 0.86, 0.809, 0.839, 0.725, 0.667, 0.87, 0.892,
            0.904, 0.82, 0.879, 0.815],
    'BERT𝐿𝑀𝑅': [0.953, 0.732, 0.88, 0.912, 0.941, 0.917, 0.906, 0.835, 0.925, 0.929, 0.89, 0.927, 0.887, 0.755, 0.92,
                 0.925, 0.944, 0.878, 0.877, 0.891],
    'GPNE': [0.242, 0.435, 0.73, 0.594, 0.141, 0.421, 0.397, 0.369, 0.479, 0.783, 0.754, 0.792, 0.664, 0.466, 0.728, 0.24,
             0.68, 0.589, 0.379, 0.52],
    'GPNE2': [0.741, 0.683, 0.214, 0.73, 0.923, 0.692, 0.738, 0.713, 0.779, 0.759, 0.817, 0.73, 0.48, 0.535, 0.76, 0.824,
              0.785, 0.757, 0.77, 0.707],
    'NTPR𝑂': [0.84, 0.718, 0.828, 0.787, 0.862, 0.654, 0.788, 0.704, 0.708, 0.885, 0.794, 0.807, 0.718, 0.553, 0.75, 0.716,
               0.772, 0.76, 0.712, 0.756],
    'NTPR𝑅': [0.92, 0.708, 0.851, 0.906, 0.915, 0.908, 0.891, 0.814, 0.881, 0.886, 0.869, 0.935, 0.863, 0.742, 0.914, 0.885,
               0.929, 0.87, 0.834, 0.869],
    'NTPR𝐹': [0.921, 0.727, 0.863, 0.896, 0.929, 0.894, 0.898, 0.801, 0.865, 0.902, 0.879, 0.929, 0.873, 0.738, 0.905, 0.897,
               0.92, 0.858, 0.849, 0.871],
    'LORE': [0.653, 0.619, 0.2, 0.711, 0.857, 0.735, 0.672, 0.651, 0.712, 0.715, 0.487, 0.694, 0.43, 0.572, 0.669, 0.472,
              0.706, 0.616, 0.587, 0.619],
    'nLORE': [0.632, 0.647, 0.167, 0.756, 0.882, 0.548, 0.798, 0.735, 0.815, 0.727, 0.737, 0.686, 0.441, 0.531, 0.702, 0.736,
               0.716, 0.722, 0.639, 0.664]
}

df_idrisi = pd.DataFrame(idrisi)