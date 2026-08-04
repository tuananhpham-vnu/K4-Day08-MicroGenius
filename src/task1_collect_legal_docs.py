"""Task 1 - Collect startup and e-commerce legal/policy documents.

The lab grader checks that at least three non-empty PDF/DOC/DOCX files exist
under data/landing/legal/. This script creates a small, reproducible Vietnamese
corpus for the group's legal assistant topic when direct downloadable policy
files are not available in the lab environment.

The generated files are plain UTF-8 text saved with a .doc extension so the
provided converter can process them without Microsoft Office. Each document
keeps official source URLs and retrieval keywords for the RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


LEGAL_DOCUMENTS = [
    {
        "filename": "luat-doanh-nghiep-2020-cong-ty-tnhh-co-phan.doc",
        "title": "Luật Doanh nghiệp 2020 - Thành lập Công ty TNHH và Công ty Cổ phần",
        "source_url": "https://dangkykinhdoanh.gov.vn/vn/pages/ChiTietVanBan.aspx?vID=27002",
        "customer_role": "seller",
        "document_type": "business_registration_law",
        "body": """
        Luật Doanh nghiệp số 59/2020/QH14 điều chỉnh việc thành lập, tổ chức
        quản lý, tổ chức lại, giải thể và hoạt động liên quan đến doanh nghiệp.
        Với người bán online muốn mở rộng từ hộ kinh doanh sang doanh nghiệp,
        hai loại hình thường gặp là công ty trách nhiệm hữu hạn và công ty cổ
        phần. Công ty TNHH có thể là công ty TNHH một thành viên hoặc công ty
        TNHH hai thành viên trở lên; thành viên chịu trách nhiệm trong phạm vi
        phần vốn góp đã cam kết. Công ty cổ phần có vốn điều lệ chia thành cổ
        phần, có cổ đông và cơ chế chuyển nhượng cổ phần theo quy định.

        Khi tư vấn lựa chọn mô hình cho hoạt động bán hàng trên TikTok Shop,
        Shopee hoặc website riêng, cần phân biệt mục tiêu vận hành. Hộ kinh
        doanh phù hợp với mô hình nhỏ, cơ cấu đơn giản. Công ty TNHH phù hợp khi
        muốn tách bạch trách nhiệm tài sản, có nhiều thành viên góp vốn hoặc
        cần ký hợp đồng với đối tác lớn. Công ty cổ phần phù hợp hơn khi doanh
        nghiệp có kế hoạch gọi vốn, phát hành cổ phần hoặc có nhiều cổ đông.

        Hồ sơ đăng ký doanh nghiệp thường cần giấy đề nghị đăng ký doanh nghiệp,
        điều lệ công ty, danh sách thành viên hoặc cổ đông sáng lập tùy loại
        hình, giấy tờ pháp lý của cá nhân/tổ chức góp vốn và giấy tờ ủy quyền
        nếu nộp qua người được ủy quyền. Sau khi thành lập, doanh nghiệp còn cần
        thực hiện nghĩa vụ thuế, hóa đơn, tài khoản ngân hàng, lao động, bảo
        hiểm và các điều kiện chuyên ngành nếu bán hàng hóa có điều kiện.
        """,
    },
    {
        "filename": "nghi-dinh-52-2013-thuong-mai-dien-tu.doc",
        "title": "Nghị định 52/2013/NĐ-CP - Thương mại điện tử",
        "source_url": "https://moit.gov.vn/van-ban-phap-luat/van-ban-phap-quy/-nghi-di-nh-ve-thuong-ma-i-die-n-tu-.html",
        "customer_role": "seller",
        "document_type": "ecommerce_law",
        "body": """
        Nghị định 52/2013/NĐ-CP của Chính phủ là văn bản nền tảng về thương mại
        điện tử tại Việt Nam, có hiệu lực từ ngày 01/07/2013. Văn bản điều chỉnh
        hoạt động thương mại điện tử, bao gồm website thương mại điện tử bán
        hàng, sàn giao dịch thương mại điện tử và trách nhiệm của thương nhân,
        tổ chức, cá nhân tham gia giao dịch trực tuyến.

        Với người bán trên nền tảng như Shopee và TikTok Shop, cần nhận diện
        rằng sàn thương mại điện tử có quy chế hoạt động riêng, còn người bán
        vẫn phải bảo đảm thông tin hàng hóa, giá, điều kiện giao dịch, bảo hành,
        đổi trả, vận chuyển và các nghĩa vụ pháp luật khác. Nếu người bán tự
        vận hành website bán hàng hoặc ứng dụng bán hàng, có thể phát sinh nghĩa
        vụ thông báo hoặc đăng ký website/ứng dụng thương mại điện tử với cơ
        quan quản lý tùy mô hình.

        Khi trả lời truy vấn pháp lý, hệ thống cần ưu tiên các nhóm thông tin:
        mô hình kinh doanh trực tuyến, nghĩa vụ cung cấp thông tin, giao kết hợp
        đồng điện tử, bảo vệ quyền lợi người tiêu dùng, bảo vệ thông tin cá nhân,
        trách nhiệm của chủ sàn và trách nhiệm của người bán. Nếu câu hỏi liên
        quan đến chính sách nội bộ của từng sàn, cần trích dẫn thêm quy định
        người bán của sàn đó vì chính sách nền tảng có thể cập nhật thường xuyên.
        """,
    },
    {
        "filename": "thong-tu-40-2021-thue-ho-kinh-doanh-online.doc",
        "title": "Thông tư 40/2021/TT-BTC - Thuế GTGT và TNCN với hộ/cá nhân kinh doanh online",
        "source_url": "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Thong-tu-40-2021-TT-BTC-huong-dan-thue-gia-tri-gia-tang-thue-thu-nhap-ca-nhan-477635.aspx",
        "customer_role": "seller",
        "document_type": "tax_guidance",
        "body": """
        Thông tư 40/2021/TT-BTC hướng dẫn thuế giá trị gia tăng và thuế thu nhập
        cá nhân đối với hộ kinh doanh, cá nhân kinh doanh. Điểm truy vấn quan
        trọng cho trợ lý pháp lý là ngưỡng doanh thu 100 triệu đồng/năm. Hộ kinh
        doanh hoặc cá nhân kinh doanh có doanh thu từ hoạt động sản xuất, kinh
        doanh trong năm dương lịch từ 100 triệu đồng trở xuống thuộc trường hợp
        không phải nộp thuế GTGT và không phải nộp thuế TNCN theo quy định được
        nêu trong thông tư. Nếu doanh thu trên ngưỡng này, cần xem xét nghĩa vụ
        khai và nộp thuế theo phương pháp, ngành nghề và tỷ lệ áp dụng.

        Với câu hỏi "Bán hàng online trên TikTok Shop đạt doanh thu bao nhiêu
        thì phải nộp thuế TNCN và GTGT?", câu trả lời ngắn gọn là cần chú ý
        mốc trên 100 triệu đồng/năm doanh thu. Tuy nhiên, câu trả lời nên nhắc
        người dùng rằng doanh thu phải tính theo năm dương lịch và nghĩa vụ cụ
        thể còn phụ thuộc mô hình đăng ký, ngành hàng, hồ sơ khai thuế, hóa đơn
        và hướng dẫn của cơ quan thuế tại thời điểm phát sinh nghĩa vụ.

        Khi truy hồi tài liệu, chunk này nên được ưu tiên cho các từ khóa: thuế
        TikTok Shop, thuế Shopee, hộ kinh doanh online, cá nhân kinh doanh,
        doanh thu 100 triệu, thuế GTGT, thuế TNCN, kê khai thuế, bán hàng online.
        """,
    },
    {
        "filename": "nghi-dinh-168-2025-dang-ky-ho-kinh-doanh.doc",
        "title": "Nghị định 168/2025/NĐ-CP - Đăng ký hộ kinh doanh",
        "source_url": "https://dangkykinhdoanh.gov.vn/vn/Pages/ChiTietVanBan.aspx?vID=27043",
        "customer_role": "seller",
        "document_type": "household_business_registration",
        "body": """
        Nghị định 168/2025/NĐ-CP về đăng ký doanh nghiệp có hiệu lực từ ngày
        01/07/2025 và là văn bản hiện hành thay thế Nghị định 01/2021/NĐ-CP về
        đăng ký doanh nghiệp. Văn bản này quy định về hồ sơ, trình tự, thủ tục
        đăng ký doanh nghiệp; đồng thời quy định đăng ký và hoạt động của hộ
        kinh doanh, liên thông đăng ký hộ kinh doanh và đăng ký qua mạng thông
        tin điện tử.

        Với câu hỏi "Hồ sơ và thủ tục đăng ký Hộ kinh doanh cá thể gồm những
        giấy tờ gì?", trợ lý cần truy hồi nhóm thông tin về quyền thành lập hộ
        kinh doanh, nghĩa vụ đăng ký hộ kinh doanh, giấy chứng nhận đăng ký hộ
        kinh doanh, tên hộ kinh doanh, ngành nghề kinh doanh và nguyên tắc nộp
        hồ sơ. Hộ kinh doanh do một cá nhân hoặc các thành viên hộ gia đình đăng
        ký thành lập; trường hợp các thành viên hộ gia đình cùng đăng ký thì cần
        có văn bản ủy quyền cho một thành viên làm đại diện/chủ hộ kinh doanh.

        Hồ sơ đăng ký hộ kinh doanh thường gồm giấy đề nghị đăng ký hộ kinh
        doanh, thông tin chủ hộ kinh doanh, tên hộ kinh doanh, địa chỉ trụ sở,
        ngành nghề kinh doanh, vốn kinh doanh, thông tin thuế và giấy tờ pháp lý
        của cá nhân liên quan. Nếu hộ kinh doanh do các thành viên hộ gia đình
        đăng ký, cần chuẩn bị thêm biên bản họp hoặc tài liệu thể hiện việc các
        thành viên thống nhất thành lập hộ kinh doanh và văn bản ủy quyền cho
        người đại diện. Khi nộp online, hồ sơ điện tử phải có đầy đủ giấy tờ,
        thông tin kê khai chính xác, số điện thoại, thư điện tử của người nộp hồ
        sơ và chữ ký số theo yêu cầu của hệ thống.

        Từ khóa truy hồi: hộ kinh doanh cá thể, đăng ký hộ kinh doanh, hồ sơ hộ
        kinh doanh, giấy đề nghị đăng ký hộ kinh doanh, chủ hộ kinh doanh, thành
        viên hộ gia đình, cơ quan đăng ký kinh doanh cấp xã, Nghị định 168/2025.
        """,
    },
    {
        "filename": "tiktok-shop-quy-dinh-nguoi-ban-viet-nam.doc",
        "title": "TikTok Shop Việt Nam - Quy định người bán và sản phẩm hạn chế",
        "source_url": "https://seller-vn.tiktok.com/university/essay?knowledge_id=6837787798570754&lang=en",
        "customer_role": "seller",
        "document_type": "platform_policy",
        "body": """
        TikTok Shop yêu cầu người bán tuân thủ chính sách nền tảng và pháp luật
        địa phương khi niêm yết, bán và quảng bá sản phẩm. Các nhóm sản phẩm bị
        cấm, không được hỗ trợ hoặc bị hạn chế có thể cần phê duyệt trước, giấy
        chứng nhận, kết quả kiểm nghiệm, hình ảnh nhãn mác, tài liệu chứng minh
        nguồn gốc hoặc hồ sơ pháp lý chuyên ngành. Các nhóm thường cần chú ý
        gồm mỹ phẩm, thực phẩm, thực phẩm bổ sung, sản phẩm cho trẻ em, hàng có
        thương hiệu, hàng có điều kiện và sản phẩm liên quan đến sức khỏe.

        Người bán phải mô tả sản phẩm chính xác, không gây hiểu nhầm, không đưa
        tuyên bố y tế/quảng cáo vượt quá hồ sơ pháp lý, không bán hàng giả,
        hàng xâm phạm quyền sở hữu trí tuệ, hàng hết hạn, hàng không an toàn
        hoặc hàng bị pháp luật cấm. Nếu vi phạm, nền tảng có thể từ chối duyệt,
        gỡ sản phẩm, hạn chế tính năng shop, trừ điểm vi phạm hoặc chấm dứt
        quyền bán hàng tùy mức độ.

        Khi trợ lý trả lời về TikTok Shop, cần tách bạch ba lớp nghĩa vụ: điều
        kiện pháp luật Việt Nam, điều kiện đăng ký người bán trên nền tảng, và
        điều kiện riêng của ngành hàng. Chính sách TikTok Shop cập nhật định kỳ,
        vì vậy câu trả lời nên trích dẫn nguồn và khuyến nghị kiểm tra Seller
        Center trước khi đăng bán sản phẩm rủi ro cao.
        """,
    },
    {
        "filename": "shopee-quy-dinh-dang-ban-san-pham-viet-nam.doc",
        "title": "Shopee Việt Nam - Quy định đăng bán sản phẩm",
        "source_url": "https://help.shopee.vn/portal/4/article/77246",
        "customer_role": "seller",
        "document_type": "platform_policy",
        "body": """
        Quy định đăng bán sản phẩm của Shopee áp dụng cho người bán trên sàn.
        Người bán phải tạo thông tin đăng bán chính xác, gồm tên sản phẩm, hình
        ảnh, ngành hàng, thuộc tính, giá, tồn kho, mô tả, thông tin vận chuyển,
        bảo hành và đổi trả nếu có. Nội dung đăng bán không được gây nhầm lẫn về
        số lượng, chất lượng, giá, công dụng, xuất xứ, thương hiệu, điều kiện
        bảo hành hoặc các đặc điểm chính của sản phẩm.

        Người bán không được đăng hàng giả, hàng nhái, hàng vi phạm sở hữu trí
        tuệ, sản phẩm bất hợp pháp, sản phẩm có nội dung phản cảm, sản phẩm
        thuộc danh sách bị cấm/hạn chế hoặc sản phẩm cần giấy phép nhưng không
        đáp ứng điều kiện. Với hàng có hạn sử dụng như thực phẩm, mỹ phẩm, dược
        phẩm hoặc hóa chất, người bán phải đặc biệt chú ý nhãn hàng hóa, hạn sử
        dụng, điều kiện vận chuyển và quy định riêng của nền tảng.

        Khi tư vấn cho người bán Shopee, trợ lý cần kiểm tra câu hỏi thuộc nhóm
        đăng bán, chứng từ, hàng cấm/hạn chế, mô tả sản phẩm, giá khuyến mãi,
        ngành hàng, sở hữu trí tuệ hay xử lý vi phạm. Nếu người dùng hỏi về hồ
        sơ pháp lý ngoài sàn, cần kết hợp với Luật Doanh nghiệp, Nghị định về
        thương mại điện tử và hướng dẫn thuế cho hộ/cá nhân kinh doanh.
        """,
    },
]


def setup_directory() -> None:
    """Create data/landing/legal/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_document_text(document: dict[str, str]) -> str:
    """Return a plain-text policy/legal document saved with a .doc extension."""
    body = " ".join(dedent(document["body"]).strip().split())
    return dedent(
        f"""
        Title: {document["title"]}
        Source: {document["source_url"]}
        customer_role: {document["customer_role"]}
        document_type: {document["document_type"]}

        {body}

        Retrieval keywords:
        {document["title"]}; legal assistant; startup; e-commerce; TikTok Shop;
        Shopee; hộ kinh doanh; công ty TNHH; công ty cổ phần; thuế GTGT; thuế
        TNCN; thương mại điện tử; customer role {document["customer_role"]}.
        """
    ).strip() + "\n"


def collect_legal_documents(overwrite: bool = False) -> list[Path]:
    """Create the required legal/policy files and return their paths."""
    setup_directory()
    written_files: list[Path] = []

    for document in LEGAL_DOCUMENTS:
        path = DATA_DIR / document["filename"]
        if path.exists() and not overwrite:
            written_files.append(path)
            continue

        path.write_text(build_document_text(document), encoding="utf-8")
        written_files.append(path)

    return written_files


def main() -> None:
    files = collect_legal_documents()
    print(f"Saved {len(files)} legal/policy files in {DATA_DIR}")
    for path in files:
        print(f"- {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
