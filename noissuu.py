# SPDX-License-Identifier: WTFPL
import typing
from io import BytesIO
from dataclasses import dataclass, fields, field, is_dataclass
from functools import partial
import struct

import fitz


# Binary files
@dataclass
class IssuuSmartZoomUnk4Unk1:
        unk1: float
        unk2: float
        unk3: float
        unk4: float

@dataclass
class IssuuSmartZoomUnk4Unk3:
        unk1: float

@dataclass
class IssuuSmartZoomUnk4:
        unk1: list[IssuuSmartZoomUnk4Unk1] = field(default_factory=[])
        unk2: bytes = None
        unk3: IssuuSmartZoomUnk4Unk3 = field(kw_only=True)

@dataclass
class IssuuSmartZoom:
        unk1: int
        unk2: float
        unk3: float
        unk4: list[IssuuSmartZoomUnk4]

@dataclass
class IssuuSmartZoomInfo:
        zooms: list[IssuuSmartZoom]

@dataclass
class IssuuText:
        text: str
        unk2: float
        unk3: bytes
        unk4: bytes

@dataclass
class IssuuTextArea:
        unk1: int
        unk2: int
        unk3: int
        unk4: list[IssuuText]

@dataclass
class IssuuTextInfo:
        texts: list[IssuuTextArea]
        build: str

@dataclass
class IssuuImageLayer:
        image_id: int = 0
        width: int = field(kw_only=True)
        height: int = field(kw_only=True)

@dataclass
class IssuuTextLayer:
        text: str
        font_id: int = 0
        size: float = field(kw_only=True)
        color: int = field(kw_only=True)
        matrix: bytes = field(kw_only=True)     
        origin_x: bytes = b''
        origin_y: bytes = b''
        scale: bytes = b''

@dataclass
class IssuuRectLayer:
        color: int
        blend: int = 1
        box: bytes = b''

@dataclass
class IssuuLineLayer:
        color: int
        blend: int = 1
        coord: bytes = b''

@dataclass
class IssuuLayer:
        image: IssuuImageLayer = None
        text:  IssuuTextLayer = None
        rect:  IssuuRectLayer = None
        line:  IssuuLineLayer = None

@dataclass
class IssuuFont:
        resource: bytes
        weight: int

        def __repr__(self) -> bytes:
                return self.__class__.__name__

@dataclass
class IssuuImage:
        resource: bytes = b''
        width: int = field(kw_only=True)
        height: int = field(kw_only=True)
        url: str = None

        def __repr__(self) -> str:
                return self.__class__.__name__

def fcolor(a):
        opacity = (a >> 24) / 0xff
        color = (
                ((a >> 16) & 0xff) / 0xff,
                ((a >> 8) & 0xff) / 0xff,
                (a & 0xff) / 0xff,
        )
        return color, opacity

def farr(a):
        return [struct.unpack('<f', a[i:i + 4])[0] for i in range(0, len(a), 4)]

# Note: this is a transformation matrix like in https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/transform#syntax
def matrix():
        return (1, 0, 0, 1, 0, 0)
def multiply_matrix(m1, m2):
        # we calculate:
        # [ a c e ] [ A C E ]   [ aA + cB    aC + cD    aE + cF + e ]
        # [ b d f ] [ B D F ] = [ bA + dB    bC + dD    bE + cF + f ]
        # [ 0 0 1 ] [ 0 0 1 ]   [ 0          0          1           ]
        return (
                m1[0] * m2[0] + m1[2] * m2[1],
                m1[1] * m2[0] + m1[3] * m2[1],
                m1[0] * m2[2] + m1[2] * m2[3],
                m1[1] * m2[2] + m1[3] * m2[3],
                m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
                m1[1] * m2[4] + m1[3] * m2[5] + m1[5],
        )
def scale_matrix(m, s):
        return multiply_matrix(m, (s, 0, 0, s, 0, 0))
def translate_matrix(m, x, y):
        return (m[0], m[1], m[2], m[3], m[4] + x, m[5] + y)
def untranslate_matrix(m):
        return (m[0], m[1], m[2], m[3], 0, 0)
def translate_point(p, m):
        # we calculate:
        # [ a c e ] [ x ]    [ ax + cy + e ]
        # [ b d f ] [ y ] =  [ bx + dy + f ]
        # [ 0 0 1 ] [ 1 ]    [ 1           ]
        return (
                m[0] * p[0] + m[2] * p[1] + m[4],
                m[1] * p[0] + m[3] * p[1] + m[5],
        )
def fitz_matrix(s):
        # convert to mupdf column-major
        return fitz.Matrix(s[0], s[2], s[1], s[3], s[4], s[5])

@dataclass
class IssuuLayerInfo:
        page_number: int
        version: int
        build: str
        width: int
        height: int
        usable: bool = False
        rmse: float = 0.0
        fallback: str = None
        layers: list[IssuuLayer] = field(default_factory=lambda: [])
        fonts: list[IssuuFont] = field(default_factory=lambda: [])
        images: list[IssuuImage] = field(default_factory=lambda: [])

        def render(self, pdf):
                page = pdf.new_page(self.page_number - 1, self.width, self.height)
                fonts = {}

                for l in self.layers:
                        if l.image:
                                image = self.images[l.image.image_id]
                                if not image.resource:
                                        if image.url:
                                                image.resource = requests.get(image.url).content
                                        else:
                                                raise ValueError('empty image')
                                page.insert_image(rect=(0, 0, l.image.width, l.image.height), stream=image.resource)
                        if l.text:
                                text = l.text

                                tmatrix = farr(text.matrix)
                                smatrix = untranslate_matrix(tmatrix)
                                origin_x = farr(text.origin_x)
                                origin_y = farr(text.origin_y)
                                scale = farr(text.scale)

                                fontname = f'issuu_{self.page_number}_{text.font_id}'
                                if text.font_id not in fonts:
                                        font = self.fonts[text.font_id]
                                        fonts[text.font_id] = page.insert_font(fontname, fontbuffer=font.resource)

                                color, opacity = fcolor(text.color)
                                for i, x in enumerate(text.text):
                                        point = translate_point((origin_x[i], origin_y[i]), tmatrix)
                                        morph = scale_matrix(smatrix, scale[i])
                                        page.insert_text(point, x,
                                                fontname=fontname, fontsize=text.size, morph=(fitz.Point(point), fitz_matrix(morph)),
                                                fill=color, fill_opacity=opacity)
                        if l.rect:
                                rect = l.rect

                                # TODO: blend
                                color, opacity = fcolor(rect.color)
                                (pos_x, pos_y, width, height) = farr(rect.box)
                                page.draw_rect((pos_x, pos_y, pos_x + width, pos_y + height),
                                        fill=color, fill_opacity=opacity, width=0)
                        if l.line:
                                line = l.line

                                # TODO: blend
                                color, opacity = fcolor(line.color)
                                (start_x, start_y, end_x, end_y) = farr(line.coord)
                                page.draw_line((start_x, start_y), (end_x, end_y),
                                        fill=color, fill_opacity=opacity)


def parse_i_int(stream, size=None, signed=False):
        x = 0
        i = 0
        bits = 0
        while True:
                b = stream.read(1)[0]
                x |= (b & 0x7F) << (i * 7)
                i += 1
                bits += 7
                if b & 0x80 == 0:
                        break
                if size is not None and bits >= size * 8:
                        raise ValueError(f'pos {stream.tell()}: oversized integer, expected {i}')
        top_bit = 1 << (7 * i - 1)
        if signed and x & top_bit:
                x -= 1 << (8 * (size or i))
        return x

parse_i_uint32 = partial(parse_i_int, size=4)
parse_i_sint32 = partial(parse_i_int, size=4, signed=True)

def parse_i_float(stream, size, le=True):
        import struct
        s = stream.read(size)
        return struct.unpack(('<' if le else '>') + {4: 'f', 8: 'd'}[size], s)[0]

parse_i_float_le = partial(parse_i_float, size=4)
parse_i_float_be = partial(parse_i_float, size=4, le=False)

def parse_i_buf(stream):
        l = parse_i_uint32(stream)
        b = stream.read(l)
        return b

def parse_i_wire(stream, len, level=0):
        d = []
        while stream.tell() < len:
                id_type = parse_i_uint32(stream)
                elem_id = id_type >> 3
                elem_type = id_type & 7
                if elem_type == 0:
                        elem_val = parse_i_uint32(stream)
                elif elem_type == 1:
                        elem_val = stream.read(8)
                elif elem_type == 2:
                        elem_val = parse_i_buf(stream)
                elif elem_type == 3:
                        elem_val = parse_i_wire(stream, len, level=level + 1)
                elif elem_type == 4:
                        break
                elif elem_type == 5:
                        elem_val = parse_i_float_le(stream)
                else:
                        raise ValueError(f'pos {stream.tell()}: invalid wire type: {elem_type}')
                d.append((elem_id, elem_val))
        return d

def type_from_wire(t, v, level=0):
        ta = typing.get_args(t)
        t = typing.get_origin(t) or t

        if issubclass(t, tuple):
                return tuple(type_from_wire(ta[-2] if tx == ... else tx, x) for (tx, x) in zip(t, v))
        if issubclass(t, list):
                return [type_from_wire(ta[0], x) for x in v]
        if isinstance(v, t):
                return v
        if issubclass(t, str) and isinstance(v, bytes):
                return v.decode('utf-8')
        if issubclass(t, bool) and isinstance(v, int):
                return bool(v)

        if is_dataclass(t):
                if isinstance(v, bytes):
                        wire = parse_i_wire(BytesIO(v), len(v), level=level)
                elif isinstance(v, list):
                        wire = v
                else:
                        raise TypeError(f'invalid compound wire type for {t}: {type(v)}')
                d = {}
                tfields = fields(t)
                for (elem_id, elem_val) in wire:
                        if elem_id > len(tfields):
                                raise TypeError(f'no mapping for type {elem_id} in {t}: {repr(elem_val)[:16]}')
                        field = tfields[elem_id - 1]
                        fta = typing.get_args(field.type)
                        ft = typing.get_origin(field.type) or field.type
                        if issubclass(ft, list):
                                if field.name not in d:
                                        d[field.name] = []
                                d[field.name].append(type_from_wire(fta[0], elem_val, level=level + 1))
                        else:
                                d[field.name] = type_from_wire(field.type, elem_val, level=level + 1)
                return t(**d)

        raise TypeError(f'dunno how to parse {t} from {v}')


# API response

@dataclass
class IssuuLayer:
        version: int
        uri: str

@dataclass
class IssuuPage:
        width: int
        height: int
        isPagePaywalled: bool
        dominantColor: str = None
        imageUri: str = None
        layersInfo: IssuuLayer = None

@dataclass
class IssuuTextInfo:
        version: int
        size: int
        uri: str

@dataclass
class IssuuDocumentV4:
        publicationId: str
        revisionId: str
        originalPublishDate: str
        isPaywallPreview: bool
        smartzoomUri: str
        pages: list[IssuuPage]
        textInfo: IssuuTextInfo

@dataclass
class IssuuLicenseConfigV3:
        customization_disable_reshare: bool = False
        customization_disable_search: bool = False
        customization_disable_share: bool = False
        customization_remove_link_below: bool = False
        customization_set_bg_color: bool = False
        customization_set_bg_image: bool = False
        customization_set_logo: bool = False
        customization_show_my_other_publications: bool = False
        download: bool = False
        embed: bool = False
        hide_ads_in_reader: bool = False
        norelated: bool = False
        norelatedmobile: bool = False
        upload_limit_mb: int = None
        upload_limit_page: int = None

@dataclass
class IssuuContentRating:
        isAdsafe: bool = False
        isSafe: bool = False
        isReviewed: bool = False
        isExplicit: bool = False

@dataclass
class IssuuReaderMetadataV3:
        title: str
        description: str
        userDisplayName: str
        access: str
        downloadable: bool
        originalFileSizeBytes: int
        contentRating: IssuuContentRating

@dataclass
class IssuuReaderConfigV3:
        licenses: IssuuLicenseConfigV3
        metadata: IssuuReaderMetadataV3

def type_from_json(t, v):
        ta = typing.get_args(t)
        t = typing.get_origin(t) or t

        if issubclass(t, tuple):
                return tuple(type_from_json(ta[-2] if tx == ... else tx, x) for (tx, x) in zip(ta, v))
        if issubclass(t, list):
                return [type_from_json(ta[0], x) for x in v]
        if isinstance(v, t):
                return v
        if is_dataclass(t):
                nd = {f.name: type_from_json(f.type, v[f.name]) for f in fields(t) if f.name in v}
                return t(**nd)

        raise TypeError(f'dunno how to parse {t}')



import json
from urllib.parse import urlparse
import requests

class IssuuFetcher:
        def __init__(self, session: requests.Session, output_dir: str = '.') -> None:
                self.session = session
                self.output_dir = output_dir

        def _get(self, _docname, _url, *args, **kwargs):
                url_parts = urlparse(_url)
                fn = url_parts.path.rstrip('/').split('/')[-1]

                output_dir = os.path.join(self.output_dir, _docname)
                os.makedirs(output_dir, exist_ok=True)
                resp = self.session.get(_url, *args, **kwargs)
                if resp.ok:
                        with open(os.path.join(output_dir, fn), 'wb') as f:
                                f.write(resp.content)
                return resp

        def _call(self, _method, *args, **kwargs):
                url = f'https://api.issuu.com/call/{_method}/{"/".join(args)}'
                return self.session.get(url, **kwargs)

        def _get_document(self, author: str, name: str) -> IssuuDocumentV4:
                fns = ['reader3_4.json']
                for fn in fns:
                        reader_resp = self._get(name, f'https://reader3.isu.pub/{author}/{name}/{fn}', headers={
                                'Referer': 'https://reader3.isu.pub/',
                        })
                        reader = reader_resp.json()
                        if reader['status'] != 0:
                                raise ValueError(f'issuu API responded status {reader["status"]} (message: {reader["error"]})')
                        if reader['version'] == 4:
                                document = type_from_json(IssuuDocumentV4, reader['document'])
                        else:
                                raise ValueError(f'issuu document version unsupported: {reader["version"]}')
                        return document

        def _get_reader_config(self, author: str, name: str) -> IssuuReaderConfigV3:
                resp = self._call('backend-reader3/dynamic', author, name)
                return type_from_json(IssuuReaderConfigV3, resp.json())

        def _get_download_url(self, id: str) -> str | None:
                resp = self._call('backend-reader3/download', id)
                if resp.ok:
                        return resp.json()['url']
                return None

        def fetch(self, uri: str = None, author: str = None, name: str = None):
                if not author or not name:
                        if not uri:
                                raise ValueError('either `uri` or `author` and `name` have to be specified')
                        url_parts = urlparse(uri)
                        if url_parts.hostname.removeprefix('www.') not in ('issuu.com', 'isu.pub'):
                                raise ValueError(f'unknown hostname {url_parts.hostname}')
                        path_parts = url_parts.path.lstrip('/').split('/')
                        if path_parts[1] != 'docs':
                                raise ValueError(f'invalid path {url_parts.path}')
                        author = path_parts[0]
                        name = path_parts[2]

                def fixup_uri(uri):
                        return 'https://' + uri.removeprefix('https://').removeprefix('http://')

                print(f'>> Fetching {name} ({author})')
                document = self._get_document(author, name)

                dl_url = self._get_download_url(document.publicationId)
                pdf_data = None
                if dl_url:
                        print(f'  Found download URL, fetching: {dl_url}')
                        pdf_data = requests.get(dl_url).content

                if pdf_data:
                        print(f'>> Saving downloaded PDF')
                        with open(os.path.join(self.output_dir, name + '.pdf'), 'wb') as f:
                                f.write(pdf_data)
                else:
                        print(f'  Rendering...')

                        self._get(name, fixup_uri(document.smartzoomUri))
                        if document.textInfo:
                                self._get(name, fixup_uri(document.textInfo.uri))

                        pdf = fitz.open()
                        last_layer_uri = None
                        last_layer_i = None
                        for i, page in enumerate(document.pages, start=1):
                                print(f'  Fetching page {i}')
                                page_image = self._get(name, fixup_uri(page.imageUri)).content

                                if page.layersInfo:
                                        last_layer_uri = page.layersInfo.uri
                                        last_layer_i = i
                                elif last_layer_uri:
                                        last_layer_uri = last_layer_uri.replace(f'{last_layer_i}.bin', f'{i}.bin')
                                        last_layer_i = i

                                layer_wire = None
                                if last_layer_uri:
                                        layer_resp = self._get(name, fixup_uri(last_layer_uri))
                                        if layer_resp.ok:
                                                layer_wire = parse_i_wire(BytesIO(layer_resp.content), len(layer_resp.content))

                                if layer_wire:
                                        layer = type_from_wire(IssuuLayerInfo, layer_wire)
                                        print(f'    Rendering...')
                                        layer.render(pdf)
                                else:
                                        print(f'    Using image...')
                                        pdf_page = pdf.new_page(i - 1, page.width, page.height)
                                        pdf_page.insert_image(rect=(0, 0, page.width, page.height), stream=page_image)

                        print('>> Saving rendered PDF')
                        pdf.save(os.path.join(self.output_dir, name + '.pdf'),
                                garbage=4, clean=True, deflate=True, deflate_images=True, deflate_fonts=True, linear=True)

if __name__ == '__main__':
        import os
        import argparse
        import requests
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument('-o', '--output-dir', help='output directory', default='.')
        parser.add_argument('URL')
        args = parser.parse_args()

        session = requests.Session()
        session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/112.0',
                'Origin': 'https://issuu.com',
        })
        fetcher = IssuuFetcher(session, args.output_dir)
        fetcher.fetch(args.URL)
        