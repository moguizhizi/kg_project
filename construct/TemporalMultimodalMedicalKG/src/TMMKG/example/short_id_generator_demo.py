from TMMKG.utils.secure_utils import short_id


def main():
    # 生成一个ID
    sid = short_id()
    print("Generated short ID:", sid)

    # 如果想一次生成多个
    print("\nGenerate 5 IDs:")
    for _ in range(5):
        print(short_id())


if __name__ == "__main__":
    main()
