CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�<     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�D    max       P���     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��
=   max       <�C�     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @E���R     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�fffff     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q            �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�`         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �'�   max       ;�`B     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,�$     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B-Ś     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��e   max       C��     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Tnk   max       C��8     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�D    max       P�5�     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�x���F     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��
=   max       <�o     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @E���R     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v���
=p     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q            �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�3�         8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?�x���F     `  d\   
         "   	      	      0                                 +               �         Y            #   	   #                                                                              ,            -                                       	   N��jNL��N�J�O]O�SN���O��N�>�PI(�O�ǞNQ�zN8WN�ǓO�h(O�k�M���O.J�O'tN���O�S)Oy8�O��O���N��P��xNk�)O�`OP���OPJ�N���O
m:P�O~�P�5�O״�O7��N���Nf��N�tO^�BO,�N!u�Nh�N=��OW�hOc�O�aOv�$N�.~NS�O�N��O<�]N�AN}
�O�k/N�@N�@�O�RnP5��N�
8O��Nߋ�OfI�N7�O���N�N�*O1�ONF�N���OUk�Nbn:N[�O, PO��N&@M�D <�C�<49X%�  �ě��ě��ě���`B�o�o�t��t��49X�49X�D���T���T���e`B�u��o��C���C���C����㼣�
���
��1��1��1��1��1��1��j��j��j��j��j��j��j��j�ě��ě��ě��ě����ͼ��ͼ�h�����C��t��t��t��t���P�#�
�'',1�,1�,1�49X�<j�<j�<j�D���D���H�9�]/�aG��ixս}󶽅����㽝�-���T���署-��
=����������������<<IU`YUOIC<<<<<<<<<<lmyz}���������zmhjll�������������������������������EN[_ghrg[NLEEEEEEEEE�����	"%#	�������������������������s��������������{vnls������		����������������������������//<<@HJKH<3///5/////��������������������aUK</#
 
#<HU\]a������(3+�������� #//72/##          �������

������������������������y{������������|{yyyy�������

������������
#()'#
����@Uenz������~znaZPH?@O[mz��������zbTNNLO��
#%#
�������������������.5Ngmssqi_[UNB51*,+.x�����������������wx����������������������������������������Y[^dhlt{{z||xtlh[VWYHKXat��������maTKHGH459BN[t��|rg[NHB<.,45B[s�����������[N-*5��������������������afjnz���������znle]a()6@BDHIB:6)!BBDO[[c[OB=6BBBBBBBBKOS[htu����|th[QOKLKSTWamoz�����zmaYTRS����������������������������������������~����������~~~~~~~~LN[\gqtytg`[WNLLLLLL~����������������xv~��������������������.0Ubpvy��{nb\UI<9,'.����������������������������������������()15;5)�������������������������


�����������
#,/1/)# 
�����pt}�������ztpnpppppp
"#*.04100-#
et������������tg`]`e�������������������������������)8BN^gt�����t_NB5$������ !%�������RUZ`antz�����zniaXUR��������������������##0:<INNKIC<70####2<CGIJRUY\\YRI<70..2OO[hmlh`[ZOOOOOOOOOO������
0<INL<0����lno{}����{nnllllllll������������������������
������v{����������������zv��������������������Z_ght����������tg`ZZ�������������������

���������!#(+,/<BHG><3/#./<CHUaagcaWUH<2//,.����������������������������������������¦²¾¿¿¿¿µ²¦�ʼɼ¼��ʼּܼ���ټּʼʼʼʼʼʼʼ�ĚĘčččĐĚĦĪĳĿ��ĿķķķĳĦĚĚ���������������
�#�/�5�6�2�4�/�#��
����ƤƚƕƔƚƥƧƳƶ������������������ƳƤƁ�{�|ƁƍƎƚơƟƛƚƎƁƁƁƁƁƁƁƁ�;�"��	���������	��"�/�;�?�H�Q�]�T�H�;�s�q�g�d�a�g�s�s�t���������������s�s�s�s���y�a�W�\�������Ŀؿ��������ݿĿ���ìàÓÇ��s�p�zÇÓàãìù��������ùì�нȽнѽݽ������ݽннннннннн��"�"�"�"�/�9�;�H�K�K�H�>�;�9�/�"�"�"�"�"�����������������������������������������O�[�h�r�m�[�]�W�O�B�6�)��!�%�%�"�)�B�O���������0�I�V�W�D�@�=�6�0�,�$��<�8�<�<�H�U�V�U�U�H�<�<�<�<�<�<�<�<�<�<��������������������������������������������������������(�-�-�)�(�"����нͽĽĽνнܽݽݽ�����ݽԽннноʾʾ־����������ʾ׾߾������	��׾�����ŹųŲŲŻ������������������������������������������
�����
����������	�����������	��#�.�;�H�K�T�P�H�;�/�������������������������
���
����������л�������'�@�r���������������r�'����O�H�B�<�B�L�O�Z�[�h�p�h�a�[�O�O�O�O�O�O�����������������6�=�C�K�T�J�C�*����ߺ��r�Y�R�[�q�~�����ɺ���F�:�!���ʺ���FcF^FJF=F1F$FFF$F+F1F;F=FJFVFcFjFxFoFc�S�G�F�S�_�l�x�������������x�l�_�S�S�S�S����׾ƾ������ʾ׾��������	�����5�(������ؿ׿ݿ���A�[�|�����s�g�N�5������߿��������"�)�(�%�������������������������*�8�)� ��9�4������ݿѿ��������������������Ŀٿ����������������������������������������������ɿ����������������������ĿƿƿĿ����������U�T�U�W�a�c�n�v�x�s�n�a�U�U�U�U�U�U�U�U�y�s�m�d�`�`�`�k�m�y�������������������y������ƳƧƧƞƧƳ���������������������u�p�h�\�[�R�U�\�a�h�uƁƁƍƏƕƎƄƁ�u���������������Ŀ������������������������B�;�;�B�G�O�[�a�_�[�R�O�B�B�B�B�B�B�B�B���
�	��	�����"�$�"����������������������������	�������	�������������������
���#�.�0�:�0�#���
���!� �#�&�-�:�F�S�_�q�x�v�x�}�n�S�F�:�-�!�[�[�n�x�{ŇŔŭŹ����������ŹŭŔŇ�n�[�{�p�o�c�o�{ǈǔǛǜǔǈ�{�{�{�{�{�{�{�{�U�U�U�a�b�n�t�zÆÁ�z�v�n�a�U�U�U�U�U�U�g�d�Z�T�W�Z�g�s���������������������s�g�����������������������������������������0�-�$����$�*�0�4�=�F�I�P�V�V�Q�I�=�0�/�*�#� �#�/�<�H�U�W�U�Q�H�<�/�/�/�/�/�/�лɻû��ûĻлܻ������������ܻл�ĳĤġĢĮ�����������
����������Ŀĳ��������ĿĿĿ�������������������������ؾ�����׾оվ׾�����	��
�	������������ؾԾ������	������	�����Y�U�_�j�������ּ���!�*�(�!���ּ��f�Y���������������������������������ƾ�������v�r�f�d�c�f�o�r�����������������������������������������ĽȽĽ�����������������ݽͽݽ�����(�4�A�M�Y�^�Z�N�A�(��ùù����ùϹٹܹ޹ܹϹùùùùùùùù��	���������������������������������	�4�(�'�%�'�4�:�@�C�C�A�@�4�4�4�4�4�4�4�4���������
�����$�)�.�$������������ùñìæçìù������������������������ù���������������Ŀѿݿ������ݿѿƿĿ����/�,�.�/�<�H�U�`�a�h�a�U�H�<�/�/�/�/�/�/ìàÔÓÎÈÇÄ�|ÇÓàìþ��������ùì������!�'�.�9�/�.�!���������l�`�l�l�m�y�}�����������������y�l�l�l�lE�E�E�E�E�E�E�E�FF$F1F=F>F1F-FE�E�E�E�E�E�E�E�E~E|E�E�E�E�E�E�E�E�E�E�E�E�E�E������!�-�0�7�-�!�������������������������������������������������� _ > J < G U U G 8 X J a W H � 3 C 4 B 4 J ? ? n k 4 B D Y T p e : W ? & X S U L ' U P f , @ S o E s Y s A B p ) 5 5 k v } C . g ) � I : " H G C 8 ~  O 6 &    h  �  �  U  �  7  �  �  n  q  v  .  �  �    �  i  �    ?  O  �  �    n    s  �    e  �    w  �  �  �  �  8  	  o  C  �  �  �  A  p  =  �  �  d  �  �  �  �  �  �  �  �  $  C  /  �  c  S  �  8  �  s  �  �  �  k  �  "  L  '  ;�`B;ě����
��w�e`B�t��u�T���aG��t��T���e`B�u��w�o���
������1���u�t���P�,1��/�'����T����;d�T���o��`B�m�h���m�h�T���t���h��P�����#�
�C���/��`B��`B�'0 Ž8Q�L�ͽ49X��w�L�ͽ�w�aG��0 Ž0 Ž�o�@��T���m�h��{�H�9��%�m�h��Q�u����T����C���7L��{���-��^5��1���T���`�Ƨ�\��B�#B&�{B  PBw�B��B��A��Bf%B*�BzTB!�!B��B �~BO�B(BFB"B�&B)��B��B�{B��A���B�ZBƿB��B��B�	B��B�B�yA�9�B�bB
9wB��BF�B�oB��B��A���BP�B�,B
�uB��B`�B��B'>�Bw}B0 B��B��Be�BݢB��B$�tB
c�ByBk�B	LKB,�$B�gB .SB%��B&�vBTzB$lB(�QB�B�B� B
�LB
�BE3BmUBK�BE�B"spBg�B@@B&��A��6B�ZB��B�~A��BM\B*EcB�7B!�.B�@B!@�B;�B8rB�{B�BHB)�_B��B�?B�'A��B��B��B�VB�B��B�B6BȔA�z�B��B	c�B��BDB��B�B�A�U�B@sB�B
ˊB��B<B=�B'[�B�B(NBCB��B?�B4	B��B$��B
@�B�pB5�B	�FB-ŚB;�B ::B%�2B&��BCOB#�	B(�BÅB*BĿB
�B
jB?�B��BE�B��B"O�BGVA��@���A��A���Bz�B�pA���A�"SAu:�A�w�A+&�A� �AJN�A�N�B
�A�s�A�A��7A*��AS��A��zA���A���A�Y�@֔�A�A�8�@,&�C��@��.AU��A�$�A�S�A�UAx�eA�)�Av�A��)Am�B��B��AuW�A�`A�\�A�[A��@��7A�IB��A�&PA�onA���B
tbAÂ�@���A�yA�RAW&AZ�@��AMMo@��2A"�0A6�#>��eA��x@���B	(A�Ay;�A�yHA�K�A��A�tC��pC�w@m'qA�T�A��(@�f)A�x+A��B��B�A�r�A���Au!Ä�A*�FA���AK�AשiB
?A�~IA��,A��	A+&AT�	A���A��ZA��/A�lj@٪�A�~�A�o�@;�bC��8@��0AWU-A��WA���A�vaAx�KA�|lAw8WA�~AmaqB�BMyAv@(A�r�A���A��A�u7@|qiA��B�LAƊ�A��eA���B
ATA���@�g�A��A��AVVUAY�AؚAMAz@깂A!�yA74>TnkA�b�@�FB��A�GAx�A�rlA�uA�jA��C��$C��@lrA�wN            "   	      	      0                                 ,               �          Y             $   	   $                                                                              ,            -                           	            	                              3                  %               %         !      ?      !   ;            -      ;   %                                                                           7            !      -                                                               '                  %                              %         /            )      ;   #                                                                           7                  -                                    N�ϸNL��N�J�O6 �O�SN���O��N�>�P��Ov��NQ�zN8WN�ǓO׀O�I�M���OL�O'tN���OinO[��O_"O(
NE9�O��'Nk�)O�a[P7��O�N��ZO
m:PQ�O~�P�5�O�ʧO'_?N���Nf��N�tO^�BO,�N!u�Nh�N=��O=h�N�!OO�aO��N�.~NS�O�N��O/5�N�AN}
�O�
�N�@N�@�N�pdP5��N�
8N��Nߋ�O#K+N7�O���N�N�I�O1�ONF�N���O��Nbn:N�O, PN���N&@M�D   �  P  3  Q  �  �    M  �  >  ;  �  D  '  �  �  �  �  I  -  o  �  �  �    a  �  �  �    ]  �  �  �  H  \  �  �  �  9  �  >  �  �  y  .  �    [  �      '  �    0  �  �  !  �  P  �  �  .  �    �    �  r     T  �  �  1  )    *<�o<49X%�  �#�
�ě��ě���`B�o���
�D���t��49X�49X�ě��e`B�T���u�u��t�����t����
��`B��1��t���1��`B�#�
��/��j��1��/��j��j�����ě���j��j��j�ě��ě��ě��ě����ͼ��������\)�C��t��t��t���P��P�#�
�0 Ž',1�L�ͽ,1�49X�D���<j�]/�D���D���H�9�e`B�aG��ixս}󶽑hs���㽟�w���T��1��-��
=������ �������������<<IU`YUOIC<<<<<<<<<<lmyz}���������zmhjll�������������������������������EN[_ghrg[NLEEEEEEEEE�����	"%#	�������������������������w}��������������~utw������������������������������������//<<@HJKH<3///5/////��������������������#/<HMPKHD<0/#!������'1)������� #//72/##          ����

������������������������������������������
�����������
!'((&#
����LUnz�������zndaYUTLL[akmz������}zmaYTT[[��	

������������������������157BN[bhgec_ZNB5/001������������������������������������������������������������Y[^dhlt{{z||xtlh[VWYN\y��������xmaTOKJKN459BN[t��|rg[NHB<.,45B[s�����������[N-*5��������������������^aglnz}��������znfa^()6@BDHIB:6)!BBDO[[c[OB=6BBBBBBBBKOS[htu����|th[QOKLKSTWamoz�����zmaYTRS����������������������������������������~����������~~~~~~~~LN[\gqtytg`[WNLLLLLL�����������������{��������������������.0Ubpvy��{nb\UI<9,'.����������������������������������������()15;5)�������������������������


�����������
#+/1/(#
�����pt}�������ztpnpppppp
"#*.04100-#
gt�����������tmga^bg�������������������������������O[]glt����tg[UOOOOOO������ !%�������RUZ`antz�����zniaXUR��������������������##0:<INNKIC<70####036<HIQUXYYVUNI<4100OO[hmlh`[ZOOOOOOOOOO������
0<INL<0����lno{}����{nnllllllll������������������������
������v{����������������zv��������������������dgst��������ytlg`_dd�������������������	 ������������!#(+,/<BHG><3/#./8<EHU_afbaVUH</-..����������������������������������������¦²½´²¦�ʼɼ¼��ʼּܼ���ټּʼʼʼʼʼʼʼ�ĚĘčččĐĚĦĪĳĿ��ĿķķķĳĦĚĚ�����������������
��#�/�1�.�0�/�#��
��ƤƚƕƔƚƥƧƳƶ������������������ƳƤƁ�{�|ƁƍƎƚơƟƛƚƎƁƁƁƁƁƁƁƁ�;�"��	���������	��"�/�;�?�H�Q�]�T�H�;�s�q�g�d�a�g�s�s�t���������������s�s�s�s�����y�k�e�f�p���������Ŀ�����ݿ�����ìàÓÎÇÃ�|�w�{ÇÓßìùü������ùì�нȽнѽݽ������ݽннннннннн��"�"�"�"�/�9�;�H�K�K�H�>�;�9�/�"�"�"�"�"�����������������������������������������6�+�)�(�)�-�1�6�B�O�S�[�f�[�Z�O�O�G�B�6�$�����	����0�I�V�V�D�?�=�5�0�)�$�<�8�<�<�H�U�V�U�U�H�<�<�<�<�<�<�<�<�<�<��������������������������������������������������������(�-�-�)�(�"����нȽǽнҽݽ�����ݽннннннно׾ʾž��������žʾ׾������ �������ŹŵųŴż�����������������������Ź����������������������
��
�	�������������	���	�
��"�)�/�;�?�F�G�;�/�"������������������
���
�����������������M�'������4�M�r��������������r�f�M�O�H�B�<�B�L�O�Z�[�h�p�h�a�[�O�O�O�O�O�O��������������������*�6�B�M�C�*�����~�j�d�j�r�������ֺ�����
���ֺ������~FJFEF=F1F$FF#F$F0F1F=FJFUFVFbFcFgFmFcFJ�S�O�S�S�_�l�x���������x�l�_�S�S�S�S�S�S����׾ƾ������ʾ׾��������	�����(����������(�A�Z�g�v���{�s�g�N�5�(������߿��������"�)�(�%�������������������������*�8�)� ��9�4��������������������������Ŀֿ�����ݿѿ����������������������������������������׿����������������������ĿƿƿĿ����������U�T�U�W�a�c�n�v�x�s�n�a�U�U�U�U�U�U�U�U�y�s�m�d�`�`�`�k�m�y�������������������y������ƳƧƧƞƧƳ���������������������u�p�h�\�[�R�U�\�a�h�uƁƁƍƏƕƎƄƁ�u���������������Ŀ������������������������B�;�;�B�G�O�[�a�_�[�R�O�B�B�B�B�B�B�B�B���
�	��	�����"�$�"����������������������������	������	���������������������
���#�*�/�#���
�����!� �#�&�-�:�F�S�_�q�x�v�x�}�n�S�F�:�-�!Ň�|�{�p�{�ŇŔŠŭŹ����������ŹŭŔŇ�{�p�o�c�o�{ǈǔǛǜǔǈ�{�{�{�{�{�{�{�{�U�U�U�a�b�n�t�zÆÁ�z�v�n�a�U�U�U�U�U�U�g�d�Z�T�W�Z�g�s���������������������s�g�����������������������������������������0�.�%����$�,�0�3�=�D�I�O�U�T�N�I�=�0�/�*�#� �#�/�<�H�U�W�U�Q�H�<�/�/�/�/�/�/�лɻû��ûĻлܻ������������ܻл�ĳĦģĦĳĿ���������
������������Ŀĳ��������ĿĿĿ�������������������������ؾ�����׾оվ׾�����	��
�	��������������������	�����	�������������Y�U�_�j�������ּ���!�*�(�!���ּ��f�Y���������������������������������ƾ�������x�r�f�r����������������������������������������������ĽȽĽ��������������(��� ��������(�4�A�B�M�S�Y�M�A�5�(�ùù����ùϹٹܹ޹ܹϹùùùùùùùù��	���������������������������������	�4�(�'�%�'�4�:�@�C�C�A�@�4�4�4�4�4�4�4�4�����������
�����$�'�)�$�����ùñìæçìù������������������������ù���������������Ŀѿݿ������ݿѿƿĿ����/�,�.�/�<�H�U�`�a�h�a�U�H�<�/�/�/�/�/�/àÛÓÓÌÎÓàìùú����������ùìàà������!�'�.�9�/�.�!���������y�n�y�������������������y�y�y�y�y�y�y�yE�E�E�E�E�E�E�E�FF$F1F=F>F1F-FE�E�E�E�E�E�E�E�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E������!�-�0�7�-�!�������������������������������������������������� X > J : G U U G ) ^ J a W I | 3 < 4 @ " K 9 7 Y A 4 K < I E p c : W > # X S U L ' U P f % 6 S ^ E s Y s @ B p 0 5 5 Z v } 6 . K ) � I 8 " H G 4 8 \  O 6 &  �  h  �  �  U  �  7  �  l    q  v  .  T  *    `  i  �  �  �  �  `  m  [  n  u  *  M  �  e  �    w  �  i  �  �  8  	  o  C  �  �  �    p  �  �  �  d  �  �  �  �  `  �  �  �  $  C  �  �  u  S  �  8  �  s  �  �  4  k  H  "  !  '    =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  s  c  S  D  3  !    �  �  �  �  �  �  q  P  0  P  E  :  /  $    	  �  �  �  �  �  �  �  �  }  s  k  c  [  3  (        �  �  �  �  �  �  �  �  �  o  M  <  .       A  K  Q  L  <    �  �  x  R    �  �  E  �  �      �  R  �  �  �  �  z  q  l  g  Z  L  ;  '    �  �  �  �  o  B    �  �  �  �  �  �  �  �  �  y  r  j  b  Z  O  E  :  0  &      	  �  �  �  �  �  �  �  �  �  �  �  �  �  v  ^  A     �  M  A  4  (    
  �  �  �  �  �  �  �  d  G  )     �   �   �  i  �  �  �  �  �  �  �  �  �  ~  Z  -  �  �  P  �  `  �   �    0  =  4  '    �  �  �  �  p  B    �  �  C    �    `  ;  ,         �  �  �  �  �  �  �  �  ~  o  ^  M  <  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  D  B  A  @  ?  <  6  0  *  $           �   �   �   �   �   �  �  �  �  �           '      �  �  f  *  �  �  ?  �  d  �  �  �  �  i  K  B  6    �  �  �  h  $  �  �     �  !   o  �  �  �  �  �  �  �  y  e  O  8       �  �  �  i  ;    �  �  �  �  �  �  {  i  R  9       �  �  �  �  �  k  J  .    �  �  �  �  �  �  �  �  �  �  z  m  a  R  =  (     �   �   �  =  C  G  I  H  F  D  @  :  .    �  �  �  x  <  �  �  X    >  �  �        ,  (      �  �  �  �  d  1  �  �  �  �  >  o  _  S  J  7       �  �  �  |  ^  @  $    �  �  �  j  �  �  �  �  �  �  �  �  ~  o  Y  6    �  �  L  �  H   �   f  �  �  �  �  �  �  �  �  �  �  �  �  �  w  J    �  ^  �  T  \  �  �  �  �  �  �  �  �  �  �  �  g  J  *    �  �  V  	  
�  F  �  (  g  �        �  e  �    
�  
7  	X  W  (  z   �  a  V  L  B  8  +        �  �  �  �  g  )  �  �  �  ]  /  L  j  �  �  �  �  �  s  ^  A    �  �  W  �  �  9  �  �    |    I  r  �  �  p  [  >    �  �  �  m  -  �  �  �  	  :  �  �  �  �  �  �  _  5    �  �  `    �  b  �  �    �                       �  �  �  �  �  �  �  �  �  U  $  ]  O  @  1       �  �  �  �  �  �  �  �  �  �  x  R  $   �  �  �  �  �  �  �  �  �  �  �  �  �  X    �  �  0  �  P  }  �  �  �  ~  u  k  ^  P  ?  -       �  �  �  �  q  S  7    �  �  �  �  �  �  �  �  �  �  �  �  �  �  F      �  D     /  D  H  @  1      �  �  �  �  K  
  �  X  �  s  �  N   �  T  Y  \  Z  N  ?  /      �  �  �  �  U  (  �  �  �  <   �  �  �  �  �  �  {  o  ]  J  7       �  �  �  �  p  L  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  �  �  ~  x  r  l  f  ]  T  K  B  8  /  %       �   �   �   �  9  6  4  2  -  &      �  �  �  {  P  %  �  �  �  Y     �  �  �  �  �  �  �  �  �  v  Y  9    �  �  �  �  �  �  �  �  >  1  $    
  �  �  �  �  �  �  �  �  {  q  h  ^  U  K  A  �  �  �  �  �  �  �  �  �  x  e  L  2    �  �  �  O     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  x  v  t  n  d  Y  O  D  8  +      �  �  �  n    �  [    '  '      �  �  �  �  �  �  p  [  B  #  �  P  �  �  z  �  �  }  l  X  B  +    �  �  �  �  �    f  >  	  �  �  5  �  �  �  �      �  �  �  �  �  �  �  �  �  �  i  E    �  [  R  H  =  1  #      �  �  �  �  �  {  V  /    �  �  y  �  �  z  t  n  g  a  h  u  �  �  �  �  �  �  �  �  {  o  b      	  �  �  �  �  �  f  X  I  5    �  �  z  9  �  �  �      �  �  �  �  �  �  �  �  ~  l  Z  J  =  /  "       �  '  '      �  �  �  {  S  -    �  �  ~  D    �  k    �  �  �  �  �  �  �  �  �  z  o  b  S  E  4  !    �  �  �  �          
  	       �  �  �  �  �  �  �  �  �  �  r  b    +  /  (      �  �  �  �  �  ~  [  0    �  �  M  �  T  �  u  g  X  F  4  "    �  �  �  �  �  v  P  )     �   �   �  �  �  �  �  �  �  �  v  a  J  0    �  �  �  J    �  �  X  �  �  �  �  �               �  �  �  l  5  �  �  �  �  �  �  �  �  �  [  &  �  �  j    �  s  s  E  �  ;  �  �   �  P  J  C  =  4    
  �  �  �  �    �  �  �  �  �  �  �  �  x  m  �  �    p  ]  E  ,    �  �  �  �  �  q  C  �  �  	  �  �  u  i  c  ^  Z  U  P  G  :  )    �  �  �  q  <    �  �  �    "  +       �  �  �  G    �  l    �  S  �  ?  �  �  �  �  �  �  �  �  �  �  ~  u  j  ^  P  A  0    �  �  �    �  �  �  �  f  5    +  +    �  �  �  W     �  �  D  �  �  �  �  �  z  p  f  X  I  9  )    	   �   �   �   �   �   �   �    	      	  �  �  x  �  u  �  Z  �  ,  v  �  I   �   �   O  �  �  }  x  p  f  [  L  <  +    	  �  �  �  �  �  |  [  9  r  Z  :        �  �  �  �  P    �  �  C  �  �  N    �     �  �  �  �  �  �  �  �  �  �  z  l  S  *  �  �  ]  �  �  �  
      T  T  L  9  "    �  �  k  "  �  g  �  ~  �  ^  �  u  e  T  A  -    �  �  �  �  u  H    �  �  t  <     �  p  t  x  |  �  �  �  �  �  �  �    x  q  j  H    �  �  �  1  
  �  �  ~  @  �  �  �  �  �  �  s    �  X  �  _  �  �    $      �  �  �  �  h  P  4    �  �  �  �  C  �  �  _              �  �  �  �  �  �  �  �  �  �  �  �  �  �  *  �  �  �  �  e  B    �  �  �  d  "  �  �  2  �  �  +  �