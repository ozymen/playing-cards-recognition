# generate qr codes to print and glue to back of playing cards
# derived from http://www.blog.pythonlibrary.org/2013/03/25/reportlab-how-to-create-barcodes-in-your-pdfs-with-python/

from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode import qr
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF

suites = ["D", "S", "H", "C"]
numbers = ["7", "8", "Q", "K", "10", "A", "9", "J"]

cardnames = [""]

for s in suites:
  for n in numbers:
      cardnames.append(s + n)

def createBarCodes():
    """
    Create barcode examples and embed in a PDF
    """
    c = canvas.Canvas("barcodes.pdf", pagesize=A4)


    # draw a QR code
    for i in range(1, 32+1):
        c.drawString(100, 100, 'Card ' + cardnames[i])

        qr_code = qr.QrCodeWidget(str(i), barLevel='H')

        bounds = qr_code.getBounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        d = Drawing(160, 160, transform=[160. / width, 0, 0, 160. / height, 0, 0])
        d.add(qr_code)
        renderPDF.draw(d, c, 15, 405)

        c.showPage()

    c.save()


if __name__ == "__main__":
    createBarCodes()


