def print_params(alpha):
    print(f"{sum(p.numel() for p in alpha.parameters()) // int(1e6)}M parameters")
    print(f"{sum(p.numel() for p in alpha.parameters()) // int(1e3)}k parameters")
    print(f"{sum(p.numel() for p in alpha.torso.parameters())} parameters: torso")
    print(
        f"{sum(p.numel() for p in alpha.policy_head.parameters()) // int(1e6)}M parameters: policy head"
    )
    print(
        f"{sum(p.numel() for p in alpha.policy_head.parameters())} parameters: policy head"
    )
    print(
        f"{sum(p.numel() for p in alpha.value_head.parameters())} parameters: value head"
    )
