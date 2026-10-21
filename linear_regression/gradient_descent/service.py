import sys, json


def get_theta(filepath: str) -> tuple[float, float]:
    try:
        with open(filepath, "r", encoding="utf-8") as  file:
            data = json.load(file)
            model = data.get("model", {})
            if not model:
                return 0.0, 0.0

            parameters = model.get("parameters", {})
            T0 = float(parameters.get("T0", 0.0))
            T1 = float(parameters.get("T1", 0.0))

            return T0, T1
    except (OSError, json.JSONDecodeError) as error:
        return 0.0, 0.0


def main(args):
    if len(args) != 1:
        print("Usage: python service.py <km>")
        return

    try:
        km = float(args[0])
        T0, T1 = get_theta("model.json")
        result = T0 + (T1 * km)

        print(f"Estimated price: ${result:.2f}")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main(sys.argv[1:])
    