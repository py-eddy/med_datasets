CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��vȴ9X       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N^i   max       P��@       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���-   max       ;��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?J=p��
   max       @F�=p��
     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @v|(�\     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @4�        max       @M            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @���           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �      max       ��o       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��3   max       B4t�       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�o       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�*j   max       C���       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C��B       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          >       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N^i   max       P`�1       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u&   max       ?�A [�7       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���-   max       ;��
       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?J=p��
   max       @F�=p��
     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @v|(�\     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @M            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�L�           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E   max         E       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_��Ft   max       ?�A [�7     �  Zh   
            3   0      
   
                  	      "                                           !      7                  #   4         .   -                                    )   >         )      !   (      #   5   N���OA�_N>a�O��P[��P��@N��NhܖN'0O�JOE�N>�N�P�N�ۼO=��NO��N��N���N^iO�*�O�ŞP<��N"�NLasN/	�N��,N�M�O0��OCŻO��O�!^OƓ�O S(Nt�NS#�O��N�ȂO��HOװ,N�|O���P%�iO�;�Nn�yO�&O�u6OO��Og�jO{l\N��ND��O�G O��N�;�P�1O���O)RUN��|P-�N4�O�hO�\`N��OLC�Op��Nx�;��
;�o;o�o��o��o��`B��`B�D���u��C���C����㼛�㼣�
��9X��9X��9X��9X��9X��9X��j��j�ě����ͼ��ͼ���������/���������o�+�C��C��\)�t���P��P������w��w�#�
�#�
�',1�49X�<j�<j�@��L�ͽL�ͽP�`�T���Y��Y��Y���%��%�����7L��C���t����-������������������������������������������������������������2<?IU^bnunldbSI<7322��8N[t�����[N5 ��X������

������ZXoz|��������������zoo����������������������������������������!#/<HTUXUOHC<;/-*#!!8<HUacmnsxnlaYHA<118��������������������./<>HPTUUUH</..+....))6=BOQT[\[YOJB61-))RUaloz��zr��znaRNSQR���������������������������#*-*"����������� ����������������������~|��������������~z����������z����������������{zlp���������������vol��'0RbgknrvrhI<0���,/:<>HKJH</),,,,,,,,��������|}������������������������������MNZ[gkhgd][NGFMMMMMMotu{��������������to06BEO[htwxplaOBADA709;?DHITalmnprsrmaRF9`dhnz����������znea`�����	
����������������������������
 
���������16@BHMOOOKB?=7621111������������������������" 
������������������������������������������������������
����������)201.+)$.256BNYgss{tg[NB) .����������������������	������fgtxy{{tpg``ffffffff������
			������� 
#),,)������'26@GO[hnpopjh`OB4)'��������������������KN[gt�������tge[VNKK����������������������

����������uz���������������ztu$)35=BNOVVX[[[NB5)'$��������������������?N[t���������g[NFB;?)5BNUbdfoog_[NB5)$%)������������������������������������������14+*#�����+/<>CGE<4/*(++++++++HUanz�������{nhaUHGHw����������������zqw~�����������~~~~~~~~��������������������#/<EHGB<5/#)$)+)))))�a�W�U�J�U�a�n�n�n�zÅÅ�z�n�a�a�a�a�a�a�������������
�#�$�/�4�<�<�<�/�#��
�����U�P�M�U�[�a�n�p�r�n�a�^�U�U�U�U�U�U�U�U�ʼ������������Ǽʼμּ���������ּʿy�v�u�m�`�R�L�A�=�T�m�����ǿӿֿѿĿ��y�𾺾��������˾̾׾��
��+�0�7�9�6�	���������������������������������������������������������������6�4�)��)�6�B�N�D�B�6�6�6�6�6�6�6�6�6�6�6�3�/�/�/�6�:�B�E�O�]�h�t�t�m�h�[�O�B�6�A�>�4�0�3�4�A�D�M�Z�f�s�w�s�o�f�b�Z�M�A�������������������ƾž������������������
�	�
�
���#�-�/�0�7�;�4�/�#��
�
�
�
�"�"������"�/�;�>�;�7�;�B�D�;�/�$�"�H�D�B�;�;�A�T�a�[�a�m�z�������z�m�a�T�H�T�T�G�D�>�G�T�T�`�b�a�`�T�T�T�T�T�T�T�T�.�"���׾ʾ����������ʾ׾���	��1�8�.àÙÓÇÆÇÓÕàìóõìèàààààà������������������	���	��������������нϽƽнݽ����ݽнннннннннп��ݿѿĿ����������Ŀѿݿ������!���������������������$�0�6�:�8�0�'�$������������Z�@�A�Z�_�s������������� ���������������������������������������������Z�X�P�Z�g�s�}�w�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�������(�-�0�(�����������V�T�I�H�=�I�V�X�b�o�q�t�o�b�V�V�V�V�V�V������ھ׾ʾ��������ʾ׾������ ������������������}�����������������������������������������������������	��
�������g�N�A�5�5�D�Z�g�r�s�������������������g�s�Z�N�5�2�2�5�A�Z�s�������������������s�:�-�"��!�-�:�F�S�_�l�����������x�l�_�:��������������������������������������׻S�M�S�U�_�l�t�x�|�������x�l�i�_�S�S�S�S�ѿ˿̿Ϳѿݿ߿��ݿѿѿѿѿѿѿѿѿѿѿ������������Ŀѿ������
����ݿѿĿ��������������ɺֺ��ֺԺɺ����������������������h�r�r�����������ֺ�����ֺɺ��������4�@�M�Y�f�q�{�{�r�f�Y�4�'��ܻػлû����ûлܻ������������ܻ������������*�6�C�J�F�G�L�P�L�C�6��f�f�r�����ʼ��,�.�!������ʼ�����f����æÓËÈÏàì���������������������0�,�&�0�<�I�U�V�U�S�I�<�0�0�0�0�0�0�0�0�������������������������������������������	��	��"�*�/�;�G�Y�g�j�f�a�T�H�/���ܹڹϹù������ùϹܹ���
����������f�Y�4�)�'��%�'�4�@�M�\�f�o�p�n�r�x�r�f¿´©­³¹¿�����������������������¿������!�-�6�:�E�B�:�-�'�!������l�h�_�V�Z�_�l�t�x���|�x�l�l�l�l�l�l�l�lƎƁ�u�\�O�6�*�6�C�\�h�uƁƚƠƩƨƦƚƎ�n�a�U�R�H�B�G�H�U�a�n�y�z�~ÃÄÁ�|�z�n�������������Ŀѿݿ߿ݿڿѿȿĿ���������������������������)�8�B�Q�O�@�)������X�L�L�T�[�tčĦĳĿ��ĿľĳĳĬĚā�h�X�6�0�)�&�"�)�-�6�B�O�[�h�k�h�a�[�R�O�B�6���������*�-�6�<�>�6�*������c�Q�H�H�`�����ݽ��������ݽĽ����y�cFFF	FFF$F1F8F1F.F$FFFFFFFFF�������������ùܹ���5�>�3������ù����������������������������
�������<�9�0�2�<�H�U�`�X�U�H�@�<�<�<�<�<�<�<�<����޼ڼڼ�����������������E�E�E�EuEkEuE�E�E�E�E�E�E�E�E�E�E�E�E�E����!�$�!������������ ������ : 8 p D f \ Z @ ^ 8 * _ / > 4 [ g = B W @ I f N 3 < n p e G ? N 8 : s R 6 R O * d K b F ; ; 3 C b ) m < f 4 t . R 2 D o C j d ? D T E    �  �  �  _  5  �  1  �  =  ^  �  �  �  �  �  A  )  �  �  +  ^  :  z  G  U  C  �  	  �  �  �  �  �  "  �  �  9  �  d  �  #  j  �  �  {  W  $  �  G  �  �  R  �  [  �  �  �  k  �  Z  �      �  �  $  ���o�49X��o��t��]/�P�`�ě���C���9X�o�49X��j������1��`B�ě��e`B�+��/�ě��t��]/�Y���������/�t����'��}�q����{��w��w���m�h�,1��hs��E�����]/��1����<j�Y���+��C���%�u�P�`�H�9��o����]/��^5��l�������+��vɽ��
�ě���
=���-���   ��E�B�B�MB�B&�<B�:BáBT�B�'B��B�B��B4t�B�^BG�B��BIB`�BߒBB�B^B*]B��B%��B��B'�B�#B�GB��B�A��3B�nBfBd�Bi�B1B	.B+�B"2�B8:Br|B�$B�dB,�$B��B	�$B�dB��BݟB!�yB	SB#	 B$D�B ��B�B�B	��B�~B�B٢B�B�<B�vB;�B
��B`�B��B��B$�B�vB17B&��B$B	�B�fB�!B�B�$B��B4�oB�BF�B�jB@
B>�B�@Bz�B9�B)�mB{1B&>oB�hB?�BfB� B�B-zA��B�B;�B;�B{gB@�B+�B>LB"?B�VB?�B�!B�B-?�B��B	D�BÂB��B>�B"B8B	L�B#(MB$EfB ��B��BP�B	;�B�>B��B�B�EB@�BBqB�UB
ĠB7.B?�B�A�G A�8lA�L2A �tAp�AW��A�˶A01zAס�A�DA=b�AN�xA�HOA�> A��Af�FAT�/A�V A�/�A*�'A~6PB�
A�{�A�:OA���A5�fB��ATRA���A���A���A�1]@��7A�ʹ@�+�A{�'Ay�2@4�@'�@��@���A��YA@AA���A���A�P�A��x>�*j@ո�A�V�@gy�@�jB}�A���Ax�%A�h>A��bA���A�ܴA�VC���?=�A��A�J�A��C�'�A
�Aǆ*A��Aş�@�Z/Aq�AW��A��A0��Aր^A؎�A<�AM~A�l�A�B�A�}Af��ASʟA�v�A���A)�&A~�!B��A��A��A�y.A6�bB�AUYA���A���A��A���@��jA��K@�?,A|�DAy!@4�@j@� �@��B  �A�A��A즱A� RA�|j>�F�@�#�A��@d�T@�~BE!A�#Az�8A��fA�gVA؜�A��kA�C��B>��A��A�3)A�SC�'bA	\   
            3   0                           	      #                                        	   !      7                  $   5         /   -                                    )   >         )      "   )      $   6                  3   7                                 #            !   %   7                        '   #   #                  '   !         7                                       )   !         -      )                                 1                                                !   1                              !                  '   !         +                                       '            !      )               N���OA�_N>a�O� O$rcP`�1N3iCNhܖN'0O�JO:�N>�N�P�N�ۼN��NN�R�NO�\N��kN^iO�C�O���P'�)N"�NLasN/	�N��,N�M�N�D
OCŻO��O"�0O��nO S(Nt�NS#�O_�N�ȂO��HOװ,N��OS\�P�O�ѨNn�yO�&OZ�OO��Og�jO{l\N.ROND��O�G O	MN�;�P�OP�O�N��|O��rN4�O�hO��N��O;��Op��Nx�  P  h  �  Q  R  �  w  �  �  �  I  H  �  &  a  h  R  N    M  -  �  %  N  p  {  [  �  \    �  �  Q  �  �  q  �  �  �  L  )  I  r  �  �  @  T  �  r  �  �  �  |  �  t  $  
�    �  �  �  %  	z  F  �  
  O;��
;�o;o�D���+�#�
�e`B��`B�D���u��9X��C����㼛�㼴9X��9X��w��j��j��9X��j���ͼ��ͼě����ͼ��ͼ������������C��#�
��P�o�+�C��\)�\)�t���P�,1�''D����w�#�
�0 Ž',1�49X�@��<j�@��P�`�L�ͽT�����P�e`B�Y���+��%��%��\)��7L��\)��t����-������������������������������������������������������������3<BIUYbmjbbUQI<84333#).5BN[_ef_[TNB55))#e�����

������tbe������������������������������������������������������������!#/<HTUXUOHC<;/-*#!!7<=HPUZaimma`UHH<877��������������������./<>HPTUUUH</..+....))6=BOQT[\[YOJB61-))QWWalnz�znlnz}znaTQ�����������������������������������������������������������������������������~z����������{~���������������}{{z���������������xqoz��
0IUbnrrofI<0��,/:<>HKJH</),,,,,,,,��������|}������������������������������MNZ[gkhgd][NGFMMMMMMotu{��������������toHOX[ahlihd^[ROKGCCHH9;?DHITalmnprsrmaRF9jnz����������znhccgj������������������������������������
 
���������16@BHMOOOKB?=7621111�����������������������
 
����������������������������������������������������
����������)+./-)($,25@CNR[clkg[NB5)$$������������������
�������fgtxy{{tpg``ffffffff������
			��������	"*+)&�����'26@GO[hnpopjh`OB4)'��������������������KN[gt�������tge[VNKK����������������������

����������uz���������������ztu%)*45>BLNUUWXNB5)(%%��������������������<BN[t���������g[NFC<35BGNRWWTNIB85.+*,33�������������������������������������������*++)�����+/<>CGE<4/*(++++++++HUanz�������{nhaUHGH��������������������~�����������~~~~~~~~��������������������#/<EHGB<5/#)$)+)))))�a�W�U�J�U�a�n�n�n�zÅÅ�z�n�a�a�a�a�a�a�������������
�#�$�/�4�<�<�<�/�#��
�����U�P�M�U�[�a�n�p�r�n�a�^�U�U�U�U�U�U�U�U�ʼ������������ʼּ���������ּʼʿ��������}�y�u�y��������������������������ʾ����ľѾѾ������'�)�'�,�)��	���� ���������������
����������������������������������������6�4�)��)�6�B�N�D�B�6�6�6�6�6�6�6�6�6�6�6�3�/�/�/�6�:�B�E�O�]�h�t�t�m�h�[�O�B�6�M�D�A�8�4�3�4�8�A�M�Z�e�f�m�g�f�\�Z�M�M�������������������ƾž������������������
�	�
�
���#�-�/�0�7�;�4�/�#��
�
�
�
�"�"������"�/�;�>�;�7�;�B�D�;�/�$�"�m�a�T�L�H�F�L�T�X�^�a�b�k�m�z������z�m�T�T�G�D�>�G�T�T�`�b�a�`�T�T�T�T�T�T�T�T�׾Ѿʾþƾʾ׾�������������׾׾׾�àÞÓÇÇÇÓÚàìòòìáàààààà�������������������	�	��	��������������нϽƽнݽ����ݽнннннннннп���ݿѿĿ��������Ŀѿݿ������������������������$�0�4�9�7�0�&�$�������������������i�^�[�r�����������������������������������������������������������Z�X�P�Z�g�s�}�w�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�������(�-�0�(�����������V�T�I�H�=�I�V�X�b�o�q�t�o�b�V�V�V�V�V�V������ھ׾ʾ��������ʾ׾������ ������������������������������������������������������������������������	��
�������A�:�9�G�Z�g�k�s�������������������g�N�A�Z�W�J�J�N�Z�\�g�s���������������~�s�g�Z�F�:�+�%�.�:�A�F�S�_�l�����������x�l�_�F��������������������������������������׻S�M�S�U�_�l�t�x�|�������x�l�i�_�S�S�S�S�ѿ˿̿Ϳѿݿ߿��ݿѿѿѿѿѿѿѿѿѿѿ������������Ŀѿݿ���������ݿѿĿ��������������ɺֺ��ֺԺɺ����������������������h�r�r�����������ֺ�����ֺɺ��������4�@�M�Y�f�q�{�{�r�f�Y�4�'����ܻлû����ûлܻ��������������6�*���������*�6�A�C�I�M�K�F�C�6�����������ʼԼ���'�+�!����ʼ�����ùìÖÒÜàìù����������������������ù�0�,�&�0�<�I�U�V�U�S�I�<�0�0�0�0�0�0�0�0�����������������������������������������/�"������"�/�;�B�H�V�d�f�c�a�T�H�/��ܹڹϹù������ùϹܹ���
����������f�Y�4�)�'��%�'�4�@�M�\�f�o�p�n�r�x�r�f¿´©­³¹¿�����������������������¿����
��!�-�5�-�&�!����������l�h�_�V�Z�_�l�t�x���|�x�l�l�l�l�l�l�l�lƎƁ�u�\�O�6�*�6�C�\�h�uƁƚƠƩƨƦƚƎ�n�d�a�U�T�H�C�H�J�U�a�n�zÀÂÀ�{�z�n�n�������������Ŀѿݿ߿ݿڿѿȿĿ�����������������������������/�6�B�P�N�?�)����[�Y�X�[�b�h�tāčĖĚĞĚęčā�t�h�[�[�6�2�)�)�%�)�1�6�B�O�[�g�e�^�[�P�O�B�6�6���������*�-�6�<�>�6�*������y�l�`�Z�d�y�����������нսĽ����������yFFF	FFF$F1F8F1F.F$FFFFFFFFF�������������ùܹ���5�>�3������ù�������������������������������
������<�9�0�2�<�H�U�`�X�U�H�@�<�<�<�<�<�<�<�<����߼ۼۼ�����������������E�E�E�EuEkEuE�E�E�E�E�E�E�E�E�E�E�E�E�E����!�$�!������������ ������ : 8 p : 9 ` Z @ ^ 8 " _ / > O [ 7 H H W A F Z N 3 < n p @ G < 1 9 : s R - R O * W 8 S B ; ; - C b ) G < f / t / ' ' D P C j N ? A T E    �  �  �  *  q  -  m  �  =  ^    �  �  �  �  A  �  j  �  +  1    @  G  U  C  �  	  �  �  �  W  �  "  �  �    �  d  �  �  �  �  3  {  W  �  �  G  �  b  R  �  7  �  �  G  4  �  R  �    P  �  �  $  �  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  P  H  @  9  1  )  !        �  �  �  �  �  �  k  4  �  -  h  ]  N  ?  0  "        �  �  �  �  �  W  .    �  �  v  �  �  �  �  �  �  �  �  �  i  ;  �  �  �  �  ~  {  �  �  �  -  L  M  E  9  )    �  �  �  �  |  X  2    �  �  =  �  S  �  $  -  $    �  �  �  �     D  R  G  1    �  �  �  N  (  �  �  �  �  �  �  �  l  -  �  �        �  �  �  >  �  j  �  "  9  G  R  ]  i  r  w  u  k  Z  G  0    �  �  �  �  6  �  �  �  �  �  �  �  �  ~  j  S  9    �  �  j  ,  �  �  T  �  �    J  y  �  �  �    "  @  _    �  �  �  $  Q  �  �  �  �  �  ~  t  h  W  E  *    �  �  g  -  �  �  [    �  �    !  7  F  F  >  5  '      �  �  �  �  ;  �  �    �    H  D  @  ;  3  *  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  a  U  F  5  #        &  $  !                   �   �   �   �   �   �   �   �   �  2  B  Q  Y  ^  ^  U  L  G  B  1    �  �  �  �  �  u  X  <  h  a  Z  S  L  E  =  6  /  (         �   �   �   �   �   �   �  v  �    *  4  >  H  K  P  R  Q  K  .  �  �  V  �  g  �    N  N  L  F  <  1  "    �  �  �  �  {  [  :    �  �  �  �  �    	              �  �  �  �  �  �  �  �  t  c  R  M  M  M  M  M  M  N  N  N  N  K  E  ?  9  3  -  &         #  *  *  %        �  �  �  �    U  (  �  �  �  =   �   �  �  �  �  �  �  �  �  w  T  +  �  �  ~  >  �  �  <  �  ;  N      !    �  �  �  v  R  /    �  �  c  :    �  �  8   �  N  L  J  H  F  D  C  A  ?  =  9  2  *  #              �  p  d  X  L  ?  ,      �  �  �  �  �  u  T  3     �   �   �  {  s  k  b  Z  R  J  A  9  1  -  .  0  1  2  3  5  6  7  8  [  K  9  %    -  1    �  �  �  �  ]  -  �  �  �  T  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  v  o  i  �     $  N  [  [  W  P  C  9  +      �  �  �  �  �  g  G      �  �  �  �  �  �  l  O  .  
  �  �  �  p  @        �  �  �  �  �  �  �  d  :  
  �  �  b  .    �  �  &  �  �  �  3  S  n  �  �  �  �  �  �  v  [  1  �  �  =  �  ;  �    :  L  Q  I  4    �  �  �  ^    �  m  
  �  c  <  �  �  {  �  �  �  �  �  �  �  �  �  �  z  o  [  G  6  &    	  �  �  �  �  �  �  �  �  �  �  r  _  J  1       �  �  �  �  �  |  q  i  a  X  P  G  :  -         �  �  �  �  �  �  a  B  "  �  �  �  |  \  8    �  �  �  M    �  �  r  :  �  �  x  p  �  �  �  �  �  �    x  q  m  j  g  k  o  h  T  @  "     �  �  �  �  h  B    �  �  K  �  �     �  Q      �  �    �  L  G  5    �  �  �  7  �  �  7  �  �  D  �  �  �    	   �  �      '  "    �  �  �  j  7  �  �  }  5  �  �  :  �  �  G  4  +  :  G  D  E  D  >  1       �  �  z  C    �  �  �  &  Y  n  \  N  C  7  $    �  �  �  e  4  �  �  /  �      {  �  �  �  �  �  �  �  �  Z  -  �  �  w  &  �  W  �  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �       �  �  �  �  �  @  .      �  �  �  �  �  {  j  F    �  �  a    �  �  U  C  L  R  Q  F  -    �  �  �  e  .  �  �  �  ,  �  1  �   �  �  �  �  {  J    ,  I  8  !  �  �  T  �  W    �  f  �  =  r  i  U  C  6  8  5  $    �  �  �  �  `  *  �  �  �  �  S  �  �    n  X  =    �  �  �  �  T    �  �  D  �  �  :   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  b  C    �  �  �  �  v  t  a  E    �  �  �  R    �  �  �  �  �  �  �  �  �  �  h  D  !  �  �  �  �  r  (  �  �  *  t  m  f  _  Y  V  b  m  y  �  �  }  u  m  e  R  <  &    �  "           �  �  �  �  ^  	  �  B  �  �  �  l  /  �  s  `  �  	  	r  	�  
S  
�  
�  
�  
p  
O  
#  	�  	�  	  m  �  �       �          �  �  �  �  |  J    �  �  ;  �  0  �  �  �  �  �  �  �  �  �  �  �  {  _  >    �  �  ?  �  �  7   �   w  �  }  ~  �  �  �  �  �  �  i  3  �  �  h  $  �  v    �  �  �  �  �  �  �  �  t  Z  ?  #    �  �  �  �  n  =  �  a  �  %    �  �  �  B     �  �  L    �  �  f    �  �  6  h  �  	Z  	?  	1  	k  	N  	,  �  �  z  )  �  k  �  �    �  �  A  x  w  F  6  '      �  �  �  �  �  �  �  �  l  P  2    �  �  �  �  �  �  �  �  J    �  |  +  �  g     �    �  �  ?  J   �  
  
  	�  	�  	�  	z  	M  	   �  	  �  �  �  &  �  �  �      �  O    �  �    X  0    �  �  �  P  O  >    �  �  K  	  �