CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ļj~��#       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �e`B   max       =       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F+��Q�     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��\(��     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @N@           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @�*�           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �H�9   max       >`A�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0k�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0A�       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?.�R   max       C���       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?,��   max       C��%       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       Pvq�       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�9XbM�   max       ?�-w1��       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �]/   max       =       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F+��Q�     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @v�Q��     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @N@           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @��           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��1&�y   max       ?�-w1��     �  Y|         2                     	   3      *            	      �               	      1   <                   .         +      v      u   Q         4      >               ?      6   %         R         2      	   4         fN��2M���P&DNO�O��O��N���OB�sO�ǕN��/PN��qO�߈N#~dN�?�M�ƝN\K&O�qxP��N��hN�z�N_+�OpJ�O|��Nr��O�=WPZ8�O�t�N���N&��OSo�O��O�GN28�M��Pvq�O <�P>��N�D�P�uP[�Ox�N�ܧO8g�O�@+P&��O*jO�HOu�mON=P�N���PْO���N��O�O�pLN#�N�O|5SOx�<N�/�O�aTO*�N��OD�νe`B���ͼT���ě����
��o:�o;�o;�`B<o<o<o<49X<49X<49X<D��<D��<T��<�t�<���<�1<�1<�9X<ě�<ě�<ě�<�h<�h<�h<�<�=o=C�=\)=\)='�='�=,1=,1=49X=8Q�=<j=H�9=H�9=L��=L��=P�`=P�`=T��=Y�=m�h=m�h=m�h=m�h=}�=��P=���=��w=���=��=� �=� �=�j=�
==�/=�������������������������������������������������������������������������������*(')-/8<<HIKORUKH</*+,/8<HUan{yusofYH<0+����������������������������������������"/;ABFSXRH;/"	T[ghtz~ztga\[[TTTTTT�����#/<BJH</#�����W[[_hrtv����~tlhc[WW#/<HZ^ZPJC</*###,/4<?D</,#########��������������������ENO[`gllg[NNEEEEEEEEFGOamz�����zpaWNF����~~������������������������������������������������������FDNO[hjqh[VOFFFFFFFF�
)5ACGDHKB53"���swplns������������zszz����������zzzzzzzzsrtz���������������s���B[hmni[OB)������������	
��������"#'0<HIIIE<70-*#"""""$"��)-16BOTSOB6)��
#/<HS]`^R</#�����
#/9AHKKH</
��WV[gggtwtsg[WWWWWWWW ##'0310#          ������ )5?DILJB5����������������������bU<��������#0?Icsub&"'*36CFOQROC63*&&&&��������������\am������������zrfa\��������������������dfgint������ttjgdddd��������������������*5=BNUTNI>5)	�����#/8-<QO</
���������������*6BA96) ��"+/;HT_aefcXTH;/����������������!).220)�������	����������6BKMA6$����)6ADDFJOOKB;���������������'!)5BENPQNNGB:5)''/-.059BN[t���ug[NB5/��������������������zsz������zzzzzzzzzzz�����
#.7<=6#
����������������������)66:64) ����������

����� #/<EGHEC</*#    
����
�������� 	

 ���������
���
�	������������������������������������������������������#�%�����(�5�N�m���������P�A�(�����������������������������������������D�D�EEEE#E*E7E7E,E*EEED�D�D�D�D�D��������������������������������z�q�q�z�����	��!�"�,�(�"���	���������������������������� ���������������������������H�T�a�m�u�u�o�a�T�H�;�/�"���!�,�4�A�H�\�`�\�\�U�O�C�6�/�/�6�C�O�R�\�\�\�\�\�\�� �(�B�[�c�a�\�P�A����������
���L�Y�Y�e�l�r�}�r�e�e�Y�Q�L�B�@�?�@�H�L�L�����������������ùñîù��������ÓÔàèàÚÓÇÀÄÇÓÓÓÓÓÓÓÓÓ�"�&�/�4�;�=�E�H�K�H�<�;�9�/�)�"����"������������������������������������������������������������������������������������������	�������������Z�Q�g�z�������y�}���˽����(�A�X�V�V�T�M�A�(���ݽ��y�#�$�/�<�>�A�>�<�0�/�,�#����"�#�#�#�#�ּ߼�����ּʼƼļɼʼռּּּּּ��
��#�#�&�$�#����
��
�
�
�
�
�
�
�
�i�s�u�v�{�s�f�Z�M�A�4�+�1�4�A�C�I�Z�e�i��(�4�A�M�Z�f�n�s�s�f�Z�N�4�(���
���{ǈǒǏǔǗǔǈ�{�y�p�t�{�{�{�{�{�{�{�{E�FFF%FJF[F`FVFJF=FE�E�E�E�E�E�E�E�E��"�.�R�H�.�����׾����������������׾�"�r�������������������y�n�f�Y�O�F�S�f�r���������������r�i�f�a�f�r������*�6�:�A�6�*����#�*�*�*�*�*�*�*�*�*�*�����������������������z�r�k�j�k�w��������"�*�/�4�8�6�1�'�"��	����������������T�`�m�y���������y�m�`�G�;�.�-�0�<�A�G�T�нݽ�����ݽѽн˽ͽнннннннм������������������������ƧƳ����������+�3�0�$�����ƳƗƆ�mƎƧ�(�4�A�M�T�Z�a�b�c�]�Z�V�M�A�4�1�(�&�'�(����Ӻ��ʺֺ��-�;�O�R�N�:�������y���������������|�y�m�j�i�k�m�t�y�y�y�yàù������������������ì��l�`�Z�S�V�nà������������������ŹŭŧŢŜŝŠŹ���߽������������������������w�l�`�`�l�s�y�����������Ŀ˿̿Ŀ�����������������������Çàìù������ýùìßÓÇ�z�n�n�v�yÅÇ�����������Ŀ����������y�w�m�`�\�[�l�y��¿��­«¡�t�N�5�"��!�*�N�t�����������	������������������������޽S�\�`�d�i�l�o�x�w�l�`�S�Q�G�B�C�E�G�O�S��#�0�7�<�>�C�C�D�<�:�0�#��
����
��������'�,�3�'���������߹����#�<�I�U�l�p�m�b�E�0�
�����������������#čĚĦĲĳĸĳĦğĚčČāččččččč�~�������źĺԺպ����~�l�Y�P�M�L�R�Y�r�~���-�:�M�Z�V�F�:�-�!������׺�����B�N�[�g�t�w�t�t�j�g�[�N�G�B�>�;�B�B�B�B�	��"�/�3�3�1�/�$�"��	�	����������	�	���������"�%�&�#��������������������� �#�������������������������������������������������{ǈǔǡǩǭǹǲǭǫǡǔǈ�{�o�c�c�g�o�{�ûлܻ������������λû������ü�"�'�4�4�@�D�I�@�@�4�+�'�������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E{EsEiEhEiEu����*�/�/�/�*������������ �����I�H�<�:�<�F�I�N�U�\�U�N�I�I�I�I�I�I�I�ID�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� H ? @ J B 1 H B G X ] . + R y P H s @ O S a S = ? i Z / + ^ ] 9 > < # 8 6 N _ # % / D A * ^ ( m 5 K _ B ? @ $ & O j Z $ > h ; , "   �    �  J  h    �  �  R  �  E  �  u  M  �  &  �    t  �  �  �      �  1    h  �  6  �  �  �  W    �  m  �  �    e  �  �  �  �  /  O  �  �  �  �  �  x  h    &  �  t    �  
      !  %  ��H�9��9X=�P��o<�j<�1;ě�<�`B<�/<�o=q��<�=Y�<u<u<e`B<��
=,1>7K�<��=�P<���=��=+<�=�t�=�E�=u=�P=\)=0 �=ix�=��
=��=�w=���=�hs>"��=D��>#�
>%=��P=e`B=���=�o=�S�=�O�=�+=��w=���=���=�O�=�`B=\=��-=�^5>!��=���=� �>$�=�l�=��>n�>o=�>`A�B:�BaBղB"��B�"B�SB�EB��A��B	8~B��B
QB|�B\�BK�B��B�sA��B!��B�IBEB�BeB
�)B
��B�B�B"�B%��B:�BY�BbB�|B	,�B%r�BYB�B$*�B0k�B��BzB,_B	�B"a�B��BgkB95B/(A���B�B�0B)XB�sB~�B~�B[�B�B��B��BURB��B��B�BjwB�OB��B?UBE`B�B"��B�%BA�BÕB�"A���B	��BBjB?nB��B�NB��B�B��A�xB!��B��BAjBL�B�B
�HB
�%B=�B�B";oB&=�B@�B�B[�BA$B	?hB%D�BB�B?�B#��B0A�BbBY�B,.�B
\B"?�BO�BF�BA�B/6A�gBĄB��B?�B��B��BAGB�]B��BЍB�B?�BilB��B�B�oB��B��A�
�?.�RA��F@��C�W�A�
�A\YTAЈ�A���B �EA�G�?Բ�A��}A�U'A��_A��A�F�A���A-��A�4xAEPA�Q�A?]A:�6B��C���AQ|@��'@��A���A���A��
Ah�A*;5A�VB�A;�O@\��Am�rA�1A�>�A�6Aua�A�|UAp,�A�	�A�gYA]�A�n?V(A��AA���@��@frA�Y�A���A�tAԥNA/��B�A@�`@�X�C��A���A��C��A�wX?,��A��@��AC�O�A��@A]	�AЃSA��B ��A�}G?���A�{�Aɗ�A�f�A���A�\A�
tA-0�A�s�A
�A��#A>�A<�3BGyC��%AP�@���@��oA�~�A�G�A�A�Aj�A*ؚA��BaA<��@X\�Am�A�|�A���A%�Au�|A�Ao��A�A�~�A�Aꄨ?0�RA�f0A�hv@��@\!�A�f)A���AӀA���A/�B��@�g�@�oTC�\A��7A�NC���         3                     	   4      +            	      �               
      1   =       	            /         +      v      u   R         4      >               @      7   &         S         2      	   4         g         ,                        +                     +   7                     #   7   !                        3      0      1   #            !   /               +      '   !                                                                     %                                             !                              3            )                                 +      #                                       N'&�M���O=��NO�O ��O���N���Nr�MOo1�N��/PDNU�^O!��N#~dN�?�M�ƝN\K&O7�O_f�N~�N��N_+�N�}AO|��Nr��O�QO���Ov�N���N&��O.�On��O�p
N28�M��Pvq�O\O�;�N�D�P*yO�� O$|uN�ܧN��1O��yO��O*jO�HOu�mO.I�P�N���O�#8O��{N��N��O~c�N#�N�O|5SOx�<N�/�O_G�O*�N��OD��    �  9      k  �  �  V    �    =  q  �  �  V  �  %    �  �  �    $  �  7  |  �  f  i  �  S  �  �  R  �  x  P    f  �  p  	�  �  �    �  �  �  �  �  8  �  �  �  h  �  f  	�  j    
�  H  �  ��]/����<D���ě��o;o:�o<u<#�
<o<49X<T��<���<49X<49X<D��<D��<���=�`B<�1<�9X<�1<�`B<ě�<ě�<�h=m�h=\)<�h<�=o=�P=��=\)=\)='�=49X=� �=,1=�C�=�7L=Y�=H�9=m�h=P�`=�1=P�`=P�`=T��=e`B=m�h=m�h=�+=u=}�=���=�j=��w=���=��=� �=� �=Ƨ�=�
==�/=��������������������������������������������������������������������������������,)(*./<HIMPRIH</,,,,-./<HUanutqpj`UH<:/-����������������������������������������"/7;>>@MOH;/$T[ghtz~ztga\[[TTTTTT������#/;@GE</����_`ght{���zth________   #'/<?HLSOHE</&# ##,/4<?D</,#########��������������������ENO[`gllg[NNEEEEEEEEQQTW_aemtvz{{zuma[UQ������������������������������������������������������������FDNO[hjqh[VOFFFFFFFF)58<95)swplns������������zszz����������zzzzzzzzvtv|��������������~v)6BR[__[TOB6%�������
�����������"#'0<HIIIE<70-*#"""""$"')*6BORQOB6)#/<JUWZSF</#����
#/7@FHIH</��WV[gggtwtsg[WWWWWWWW ##'0310#          ������ )5?DILJB5�����������������������������
#+./,#
���&"'*36CFOQROC63*&&&&������
��������wrsz��������������zw��������������������dfgint������ttjgdddd��������������������)5;?ANNG;5) 
#,/275/&#
������������*6BA96) ��"+/;HT_aefcXTH;/�������	����������!).220)�������	����������6BGHF?=6+��)6?CBENNJB>6"���������������' #)5BDNOPNMCB?5)''225AN[gv����ylg[NB62��������������������zsz������zzzzzzzzzzz�����
#.7<=6#
����������������������)66:64) �����������

����� #/<EGHEC</*#    
����
�������� 	

 ����������
��
������������������������������������������������������(�5�A�N�T�Z�b�g�j�l�j�g�Z�N�A�7�)�#�&�(����������������������������������������D�D�EEEE E*E"EEED�D�D�D�D�D�D�D�D������������������������������|�x�x�z�������	��!�"�,�(�"���	�����������������������������������������������������������H�T�a�m�n�s�r�m�h�a�T�H�;�/�'�"�'�/�8�H�\�`�\�\�U�O�C�6�/�/�6�C�O�R�\�\�\�\�\�\���(�A�Q�Y�`�^�Y�N�A�������	���L�Y�e�f�m�e�Z�Y�X�L�F�D�L�L�L�L�L�L�L�L��������������������������������ÓÔàèàÚÓÇÀÄÇÓÓÓÓÓÓÓÓÓ�"�&�/�4�;�=�E�H�K�H�<�;�9�/�)�"����"�������������������������������������������������������������������������������������������������������������������������нݽ�������������ݽ˽����Ľ̽��#�/�<�<�@�=�<�/�#� ���#�#�#�#�#�#�#�#�ּ�����ּʼǼżʼʼּּּּּּּ��
��#�#�&�$�#����
��
�
�
�
�
�
�
�
�Z�f�n�o�m�j�f�Z�M�I�F�I�M�Q�Z�Z�Z�Z�Z�Z��(�4�A�M�Z�f�n�s�s�f�Z�N�4�(���
���{ǈǒǏǔǗǔǈ�{�y�p�t�{�{�{�{�{�{�{�{E�FFF$F2FJFRFJFEF=FE�E�E�E�E�E�E�E�E��ʾ׾����������׾ʾ����������������ʼ����������������v�r�g�Y�W�Q�Y�]�f�r����������������r�i�f�a�f�r������*�6�:�A�6�*����#�*�*�*�*�*�*�*�*�*�*�����������������������z�u�m�l�n�z�������	��"�+�/�2�2�-�"���	���������������	�`�m�y����������y�m�`�T�G�8�0�3�;�G�T�`�нݽ�����ݽѽн˽ͽнннннннм������������������������ƧƳ����������+�3�0�$�����ƳƗƆ�mƎƧ�4�A�M�Q�Z�_�a�a�Z�M�A�4�3�*�)�-�4�4�4�4�����!�/�:�=�<�4�!�������ںֺغ�y���������������|�y�m�j�i�k�m�t�y�y�y�y����������������ùìÓÁ�w�v�r�t�Óô��Ź����������������������ŹŲūŨũŬŵŹ�����������������������y�v�n�s�y�z���������������Ŀ˿̿Ŀ�����������������������Óàìùÿ��ùøìàÕÓÊÇ�~�~ÇÎÓÓ���������¿����������y�m�`�]�]�d�m�y�����B�N�[�g�t�x�v�t�n�g�a�[�N�B�;�5�4�4�@�B�����������	������������������������޽S�\�`�d�i�l�o�x�w�l�`�S�Q�G�B�C�E�G�O�S��#�0�7�<�>�C�C�D�<�:�0�#��
����
���������(�,�'��������������#�<�I�U�l�p�m�b�E�0�
�����������������#čĚĦĲĳĸĳĦğĚčČāččččččč�r�~���������˺Ⱥ��������~�s�e�W�R�S�[�r���-�:�K�X�T�F�:�-�!�����ۺ������B�N�[�g�t�w�t�t�j�g�[�N�G�B�>�;�B�B�B�B�	��"�/�2�2�0�/�#�"���	����������	�	������!�!� ������������������������� �#�������������������������������������������������{ǈǔǡǩǭǹǲǭǫǡǔǈ�{�o�c�c�g�o�{�ûлܻ������������λû������ü�"�'�4�4�@�D�I�@�@�4�+�'�������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EuEqElEuE�����*�/�/�/�*������������ �����I�H�<�:�<�F�I�N�U�\�U�N�I�I�I�I�I�I�I�ID�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� F ? @ J 2 . H $ E X S 3 ( R y P H V ) K I a O = ? e 5 * + ^ d * ? < # 8 1   _ " # * D ; - ( ( m 5 G _ B @ = $ $ A j Z $ > h 5 , "   M    �  J  !  1  �  |  �  �  �  s  Z  M  �  &  �  �  �  �  �  �      �  �  N  �  �  6  �  �  g  W    �  5  R  �  �  K  f  �    T  V  O  �  �  |  �  �  �  \        t    �  
    �  !  %  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �            �  �  �  �  �  �  �  �  �  u  g  Y  K  �  �  �  �  �  �  �  �  �  }  k  Y  G  2     �   �   �   �   �  �  �  $  Q  �  �       /  8  8  0    �  �  7  �    
   �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �    P    �  �  h    �  ,  �  :  S  d  h  \  K  9  %    �  �  �  �  �  g  =    �     �  �  �  �  �  �  �  �  �  �  �  y  n  ^  J  7  #     �   �   �  %  O  n  �  �  �  �  �  �  �  �  �  �  �  �  d  (  �  �  M  &  E  S  U  N  @  *    �  �  �  �  �  ^  -  �  �  �  A  �       �  �  �  �  �  �  �  �  k  T  =  (    �  �  �  �  �  �  �  �  �  �  z  `  @    �  �  >  �  �  f  `  -  �  B    k  �  �  	    #  '  %      �  �  b    �  D  �  n  �  �  Y  �  �        .  =  9  1    �  �  �  F  �    �  	    q  j  c  ]  V  O  G  ?  6  .  %        �  �  �  �  �  �  �  �  �  �  |  u  f  X  I  :  +       �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  h  Z  K  =  .     V  L  B  8  -  "    	  �  �  �  �  �  a  ;    �  �  �  �  Z  a  �  �  �  �  �  �  �  �  �  �  \  )  �  �  o  "  �  J    �  	#  	�  
8  
�  
�  b  �  �    %    �  �  
�  	�  t  �  ^  �            	  �  �  �  �  �  `  .  �  �  �  P    �  �  �  �  �  �  �  �  �  f  C    �  |    �  i    �  O  �  �  �  �  �  �  �  �  �  �  �  v  h  Z  K  =  /  !       �    1  d  �  �  �  �  �  �  �  �  �  }  f  C    �  l  �  n      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  $    
  �  �  �  �  �  �  �  �  �  v  n  s  x  z  k  \  N  U  |  �  �  �  �  l  G    �  �  o    x  �  $  �  �  �  �  m  �  �  �  �  �  �    )  7  1    �  �  ]  �  ]  �  �  O  W  i  v  z  {  u  p  g  Z  J  2    �  �  w  0  �  �  8  �  �  �  �  �  �  �  �  t  c  R  @  -    6  L  Q  P  P  O  N  f  _  X  R  M  N  O  P  Q  T  V  X  X  U  R  O  K  F  A  <  a  e  g  i  e  ]  R  E  3      �  �  �  �  �  [  0  �  -  �  �  �  �  �  �  �  �  �  �  �  �  �  e  @    �  �  [  j  4  K  R  9    �  �  �  U    �  �  H  �  p    �  g    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  d  T  C  3      �  �  �  �  Q     �   �  R  L  <  '    �  �  �  �  X     �  �  M  �  �    �  �  s  �  �  �  �  �  �  �  ~  g  H    �  �    �  B  �  �  /  u  	t  	~  	�  
T  
�  (  \  x  o  V  0  
�  
�  
M  	�  	S  �  [  �  �  P  C  7  +  "        �  �  �  �  �  �  y  Y  <  -      
�  
�  
�  
�    
�  
�  
�  
�  
�  
�  
�  
�  
�  
=  	�  �  }  �  c  
c  
�    8  W  e  _  K  *    
�  
�  
K  	�  	S  �  �  �      G  j  �  �  �  �  �  �  �  �  x  f  J  "  �  �  T  �  f  �  p  h  `  Y  Q  J  B  9  1  %      �  �  �  �  �  �  �  �  	,  	�  	�  	�  	�  	�  	n  	D  	  �  u  
  �    �    b  �    k  �  �  �  �  r  _  I  0    �  �  �  �  f  =    �  �  �  6  �    Y  !    \  �  )  [  v  �    Z    �  h  �  �  �  �          �  �  �  �  �  p  T  7      �  �  x  M  �  �  �  �  {  k  T  ;    �  �  �  i  =    �  �  �  S    �  �  �  �  �  �  �  �  j  N  -  �  �  �  ;  �  �  P  �  y  �  �  u  �  �  ~  n  \  K  <  6  2  !  �  �  K  �  j  �  k  �  y  �  �  �  �  �  �  �  �  z  N  %  �  �  s  "  �      �  �  �  �  �  s  U  5    �  �  �  �  �  �  g  <    �  �  �  n    +  6  7  #    �  �  �  a  8  �  �  P  �  �    �    �  �  �  �  �  �  u  M    �  �  Q    �  U  %  �  �  =  .  �  �  �  �  �  �  �  �  u  c  N  7    �  �  �  o  ]  `  p  �  �  �  �  �  r  _  E  #  �  �  �  z  H    �  �  �  T  %  S  �  B  a  h  g  a  V  <    �  �  2  �  E  
�  	�  �    v    �  �  �  �  �  �  w  g  W  G  7  '      �  �  �  �  �  �  f  Q  ;  $  	  �  �  �  �  �  �  k  r  x  �  �  �  �  \  7  	�  	�  	f  	B  	%  	  �  �  l  *  �  �  "  �  E  �  �  �  �  |  j  g  g  d  Z  D  '    �  �  z  6  �  �  @  �  �  $  �      �  �  �  �  �  �  r  e  Y  K  <  +    �  �  �  �  .  �  
�  
�  
�  
�  
�  
�  
�  
^  
%  	�  	�  	M  �  h  �    H  y  �     H    �  �  �  o  B    �  �  �  P    �  �  =  �  �  �    �  �  �  f  J  .    �  �  �  |  S  &  �  �  �  m  =     �  �  �  �  �  z  E     �  0  �  �  M  �  t    r  
<  �  �  �