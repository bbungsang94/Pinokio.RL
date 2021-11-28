from .basic_controller import BasicMAC
from .separate_controller import SeparateMAC
from .rode_controller import RODEMAC
from .noise_controller import NoiseMAC

REGISTRY = {"basic_mac": BasicMAC, "separate_mac": SeparateMAC, "rode_mac": RODEMAC, "noise_mac": NoiseMAC}