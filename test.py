try:
    from vision.templates import rank_templates, suit_templates
except ImportError as exc:
    print(f"Import failed: {exc}")
else:
    if rank_templates and suit_templates:
        print(f"{len(rank_templates)} rank templates and "
              f"{len(suit_templates)} suit templates loaded.")
    else:
        print("One of the template lists is empty.")
