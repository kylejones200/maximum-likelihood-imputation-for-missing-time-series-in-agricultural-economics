import logging

from ag_em_impute.runner import ensure_inputs, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    ensure_inputs()
    output_dir = run(validate=True)
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
