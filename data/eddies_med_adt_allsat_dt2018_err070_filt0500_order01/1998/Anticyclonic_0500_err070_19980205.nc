CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��Q��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�H�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��o   max       >%      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F���
=q     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @A���P     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @��          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �}�   max       >-V      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�g�   max       B3`|      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��U   max       B37�      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =ã�   max       B��      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�a1   max       B��      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          x      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�H�   max       PQnd      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U�=�L   max       ?��A [�      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >%      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F���
=q     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @A���P     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @��          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?��A [�        R<               x   1      	            O   f      r               l   j            @            7         )      "         	            "      j               &   	      !         
   "   	      +M�O( M�H�O���P���P:qN�%}NIхN�EOO��N!KWPB�P�TqO;E,P�zOlZ�OE{�N��N��P]ׯPs�OLl>N`�N2}PpnO2ĄOiSN[�P�}O��NN�PO��@N�O1��O�JN�0<N"�N2BNi�mNN�FO�lO���PI�O+��O���N\5SN�J�O�	N�ӰN�OK�jO��N5l�N��LO(�Ng[Na�kOo����o�u�T�����
��o�D��:�o;o;��
<o<t�<T��<�o<�o<�o<�o<�o<�t�<���<��
<��
<��
<�1<�1<�j<ě�<ě�<���<���<���<�/<�`B<��=��=8Q�=<j=L��=P�`=P�`=aG�=m�h=m�h=q��=q��=�%=�o=��=��=��=�C�=��=��=��-=���=ȴ9=���=��>%����������������������������������������###.02<A<50#########����������������������������5%������=@HIO[t��������thOD=��������������������DNP[gkkg[NDDDDDDDDDD'')046=>>65)'''''''',-09<BN[egkmig`[NB5,��������������������
	
6B[[cXaPIB)
����5N[g���tT)������������������5BNVafd\B)��~~�����������������~!"#<IUUYWQMID<0+!fb`bgt����ytmgffffffyprz��������}zyyyyyy����&,(��������5NgtztgN5������������������������������������������}���������}}}}}}}}}}�����
!!������

#/343/#
!#/<HPU[_UTH<2/+#!!59<HKSIHG<5555555555:659Uz���������zaUJ:��������������������8BOR[hhhf\[OKB888888�
#/<HOTMH?/#
,/0:;CHQTUUTHG>;0/,,YX^ht}���������thd[Y���������������������������������������������


������������������������������wx{}���������{wwwwww)450)��������	�������TRW^n�����������ub\Tyy����������������y���������������������������.--/;<HQNHC<4/......!"/;?B>;/"!!!!!!!!EFGLQTamz��zm]VTMME


#%0//#






��������������������spnot������������~tsqmlqt|����������ytq69<HUX_UH<6666666666���
#$$#
�����]\]akmz|������|zmea]�����������������������������������������������������������������������������������������������������ûʻʻȻ����������|�~�����������������ýнݽ޽����ݽԽннннннннннл�����'�@�F�M�Q�G�4�'����������s�������	����������N�(������A�Z�s�A�Z��������������f�A�(�����������A������������������������ź�������������������������������������������������������	���"�/�3�/�"��	�	��	�	�	�	�	�	�	�	�b�n�{ŇŔŖŗŔŇń�{�n�b�^�U�Q�P�U�_�b�M�Z�a�f�g�f�Z�M�I�H�M�M�M�M�M�M�M�M�M�M�@�Y�r�������������r�M�4����������@�y�����ÿѿҿ¿��������y�\�M�;�4�9�G�`�y���*�5�6�:�<�6�1�/�*����������
���������#�!����Ƨ�h�\�O�6�"�*�C�uƚ���ѿݿ������������ݿѿĿ������ÿѼ��������������������������y�r�k�k�r��6�B�O�[�d�g�\�[�R�O�B�7�6�+�6�6�6�6�6�6�������$�"���������������������������#�Q�]�U�I�#�
����ĿĦĎāčġĤľ������B�[�s�n�T�3�$���������������������àìñùû��������ùàÓÇ�~�y�ÇÓÛà�������ùϹйϹù������������������������/�8�<�D�=�<�/�'�(�.�/�/�/�/�/�/�/�/�/�/�����'�/�B�P�N�5�)������������������꾥�����ʾ׾��������׾ʾ���������������������	���������������������zÇÉËÇ�z�n�m�n�u�z�z�z�z�z�z�z�z�z�z�5�A�N�Z�j�t�w�������g�Z�N�F�(�����5����������������������������������������H�L�R�U�Z�V�U�P�H�<�;�6�<�C�H�H�H�H�H�H�`�m�y�}����������y�m�T�M�G�A�>�D�G�S�`�/�7�;�H�I�H�G�H�H�H�;�;�0�/�"��"�%�/�/�S�_�l�w�w�l�_�O�F�:�3�-�+�3�2�3�7�:�F�S�s���������������s�f�d�b�Z�Y�Z�\�f�n�s�z�������������������~�z�x�y�z�z�z�z�z�z�[�h�tāćā�y�t�h�g�[�W�[�[�[�[�[�[�[�[��)�*�3�2�)�(������������������������������������麋�����������������������������������������������������������������������������������ʾ׾��������׾ʾ����������������������������������x�_�S�F�6�.�/�:�S�l����!�-�:�F�S�V�P�F�D�:�-�!��������� ����������������������������y�t�o�s�y����������������� ���������������������������
��#�.�+�'�#��
���	�
�
�
�
�
�
�
�
�[�h�tčĚĝģĢĝĚčā�l�Y�O�B�?�B�O�[�����������������������������������������G�T�`�f�`�_�T�G�C�A�G�G�G�G�G�G�G�G�G�G���������������������������������~���������	��"�(�/�1�+�"��	�����������������������������������������ǡǪǭǵǭǭǢǡǔǏǈǆǈǎǔǞǡǡǡǡŭŹ������������������ŹŭũŠŞŜŠťŭ�����������������������x�u�x�������������a�d�h�a�\�U�H�<�4�<�>�H�U�Z�a�a�a�a�a�a���ʼּ������������ּмƼ������� � U K ' E C S ] b  > O A * R Z ' I H V = 3 h X D M L C Z t v ) U V B U M M S   F W . O K ] 2 I > 6 2 B \ >  s W V  �  c  )  [  L  a  �  �  \  �  /  �  �  �  �  �  �  �  �      �  X  _  {  �  Y  -  S  �  �  $  �  �  H  �  V  @  �  \  C  �  �  �  �  �  �  B  �  #  �  ]  }  �  e  �  �  �}�%   �t�<�1=�=8Q�<o<49X<o=\)<D��=�j=��<�h>�=#�
<�`B=�P<�j>o>   =L��<�`B<���=� �=�P=D��<�h=��T=��=o=�O�=�P=�hs=T��=m�h=q��=ix�=y�#=y�#=�j=�9X>(��=��
=��=�C�=���=��=��P=��=��=�j=��T=�E�>+=�/>�>-VB�B"��B%��B!K�B6B�B2;B�;B�lBZ�B��BNBpB�)B%zB?�B&!4B	�{BȜB�B�B!�B!$BG�BݱBO�B�fBRB��B��B��B�A��cBNEB�wB[�BGgB��B)B�BB�B3`|B.`B�1B,o�B��A�g�A��yB$��B!�B
�/B
^B;HB�xA�B,�B?9Bg#BB"�B%��B!��B�B@BCfB��B�Bz�B��B?�B��BITB8?B��B%��B	��B�B�BF%B"G�B?�BB�KB�IB2bB9�B��B��B�-B@�A���B�+B�AB?�B@HB��B)B}B�vB��B37�B>�B��B,��B~�A��UA�}�B%>B!�B
�4B
@>B)�BѮA�5B,��BNB�VA �`@�A*f@��
A�2�A9��A��A���A���A��wA>��@��~Ao�A���BS�A~B@��AٖCA�n�A���AԄAˬ#=ã�A�A�X�AQ_CA�aBAȦ�A�%�A��A���AjA��@�ӫAC~A��@A�A�8A/W(@��A�xnAP�@�@J@nOA �EA��A��`A�>�AJ&�AgCA�՝A��fA1J�B��A���@�kiA��A�A!/�@�,�A+ �@�@�A�R�A8�jA��qA��~A���A���A>�@վ\Ap�zA���BzDA�tR@�9A٦;AԆ}A�~[A�:�A�Ȱ=�a1A�A���AQ�WAҁ]Aȃ-A��!A�_5A�� Aj�;A�`�@��(AC:A��FA�A��/A/a�@?�A��AQj@��@mn�A�6A���A�/A݃\AJ�Af�oA��XA�&A1��B��A�r�@��A�~NA��               x   1      
            P   g      r               m   j            @            8         *      "         
            #      k                '   
      "            #   	      ,               I   1                  1   9      ;               7   3            '            +                                       !   '                                                            )                              1               '   )                                                               !   !                                             M�Nc�M�H�O��uP,SSOS��N�%}NIхN�EO�XN!KWO�O��<N쉬PQndOlZ�O07)Nv �NLn�O�PHO&�+N`�N2}O�w�O2ĄOO�N[�OȞdO��NN�POP�N�O?�O�JN�0<N"�N2BNi�mNN�FO�lO���O�w�OJzO���N\5SN�J�O.3�N�ӰN�O)`AO��N5l�N��LO(�Ng[Na�kOo��  5  S  8  �  	p  d  D  �  2    +  D  	  �  
  N  �  �  F  
�    $  �    p  Z  �  �  )  F  �  �  �  �  9  �  K  N  �  �  �  '  ,  �  v  \  �  �  n  �  �  �  �  q  	  �  N  	���o�t��T����o=,1<ě�:�o;o;��
<u<t�=e`B=�o<���=0 �<�o<�C�<��
<��
=Y�=@�<ě�<�1<�1=8Q�<ě�<���<���=#�
<���<�/=t�<��='�=8Q�=<j=L��=P�`=P�`=aG�=m�h=m�h=� �=}�=�%=�o=��=�t�=��=�C�=��-=��P=��-=���=ȴ9=���=��>%����������������������������������������###.02<A<50#########������������������������������ �������YWVWZdht{������}th^Y��������������������DNP[gkkg[NDDDDDDDDDD'')046=>>65)''''''''<89BBN[aggige[WNLB<<��������������������)56=@@?<6)�����)5@B=5)����������	
����������)5DLZ[XB5)�~~�����������������~"!#%0<ISUUPLIC<0-"dacgtxtgddddddddddyvz��������zyyyyyyyy������������ ���)3Nckg[N) ����������������������������������������}���������}}}}}}}}}}�������
�����

#/343/#
"#/<HOUZ^USH<3/+#""59<HKSIHG<5555555555B>>HUaz��������zaUHB��������������������8BOR[hhhf\[OKB888888

#/<GHNMIH</#
,/0:;CHQTUUTHG>;0/,,ZY[_ht|�������thg][Z���������������������������������������������


������������������������������wx{}���������{wwwwww)450)��������	�������TRW^n�����������ub\T���������������������������������������������.--/;<HQNHC<4/......!"/;?B>;/"!!!!!!!!MLNPTWamzz��{zsmeaTM


#%0//#






��������������������rppt}������������utrrnmqt~����������ztrr69<HUX_UH<6666666666���
#$$#
�����]\]akmz|������|zmea]���������������������������������������������������������������������������������������������������������������������������������������������нݽ޽����ݽԽннннннннннл����'�4�@�H�K�A�4�'�����������g���������������������g�Z�N�H�?�<�A�L�g�(�4�A�M�Z�f�n�i�Z�Q�A�4�(�������(������������������������ź�������������������������������������������������������	���"�/�3�/�"��	�	��	�	�	�	�	�	�	�	�b�n�{ŇŇŐŎŇ�{�{�n�h�b�V�U�T�U�^�b�b�M�Z�a�f�g�f�Z�M�I�H�M�M�M�M�M�M�M�M�M�M�4�@�M�Y�f�k�m�f�e�Y�M�@�4�'� ��'�*�4�4�y�������������������y�m�`�X�S�U�Z�b�m�y���*�1�6�7�7�6�,�*���
���������������������ƚƁ�h�O�K�U�hƁƈƞ���ѿݿ������������ݿѿĿ������ÿѼ��������������������������{�r�l�m�r��B�O�[�c�f�[�O�B�:�8�B�B�B�B�B�B�B�B�B�B�����!� ����� ����������������
�#�:�B�:�0������ĿĻıĶĽ����������)�B�M�V�T�M�B�"�������������������Óàìù����������ùèàÓÇÂ�|ÂÇÊÓ�������ùϹйϹù������������������������/�8�<�D�=�<�/�'�(�.�/�/�/�/�/�/�/�/�/�/���������#�"��������������������޾������ʾ׾��������׾ʾ������������������������������������������zÇÉËÇ�z�n�m�n�u�z�z�z�z�z�z�z�z�z�z�5�A�N�Z�b�f�k�l�i�g�Z�N�5�(�"����*�5����������������������������������������H�L�R�U�Z�V�U�P�H�<�;�6�<�C�H�H�H�H�H�H�`�m�r�y�������y�v�m�`�T�R�J�F�D�G�K�T�`�/�7�;�H�I�H�G�H�H�H�;�;�0�/�"��"�%�/�/�S�_�d�l�u�u�l�_�S�L�F�:�7�5�5�9�:�F�N�S�s���������������s�f�d�b�Z�Y�Z�\�f�n�s�z�������������������~�z�x�y�z�z�z�z�z�z�[�h�tāćā�y�t�h�g�[�W�[�[�[�[�[�[�[�[��)�*�3�2�)�(������������������������������������麋�����������������������������������������������������������������������������������ʾ׾��������׾ʾ����������������x���������������x�l�_�S�I�A�9�?�F�S�_�x��!�-�:�F�I�M�F�@�:�-�!��������������������������������������y�t�o�s�y����������������� ���������������������������
��#�.�+�'�#��
���	�
�
�
�
�
�
�
�
�h�tāčĖĚğĞĚĚčā�t�r�h�_�[�X�[�h�����������������������������������������G�T�`�f�`�_�T�G�C�A�G�G�G�G�G�G�G�G�G�G�������������������������������������������	��"�'�/�0�/�*�"��	���������������������������������������ǡǪǭǵǭǭǢǡǔǏǈǆǈǎǔǞǡǡǡǡŭŹ������������������ŹŭũŠŞŜŠťŭ�����������������������x�u�x�������������a�d�h�a�\�U�H�<�4�<�>�H�U�Z�a�a�a�a�a�a���ʼּ������������ּмƼ������� � + K 1 & 8 S ] b  > 4 (   V Z  8 L P 9 0 h X 5 M K C H t v ( U = B U M M S   F W  G K ] 2 , > 6 * 6 \ >  s W V  �  p  )    �  �  �  �  \    /  O  �    �  �  p  x  m  W  �  h  X  _    �  H  -  �  �  �  �  �  U  H  �  V  @  �  \  C  �  �  @  �  �  �  p  �  #  i  2  }  �  e  �  �    D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  5  ?  H  Q  Z  c  l  u  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �     /  ?  K  Q  O  C  1      �  �  �  �  �  }  T  8  &      �  �  �  �  �  �  �  �  w  h  Y  I  8  (      �  �  �  �  �  �  �  �  �  �  �  �  u  U  3    �  �  �  �    �  �  +  �  	  	Q  	n  	f  	I  	!  �  �  n  �  l  �  n  �    �  �  �      .  I  [  b  c  b  c  ]  C    �  _  �  o  �  D  ;  2  )      
  �  �  �  �  �  �  \  B  4  %      �  �  �  �  �  y  ^  B  &  	  �  �  �  S  	  �  g    �  t  !  2  .  +  '  #              �   �   �   �   �   �   �   �   �   �  �  �  �  �      
  �  �  �  �  �  o  9    �  �  -  �  ?  +  *  (  '  %  $  "  "  "  "  "  "  "  "  #  #  $  %  &  &  �  �  �      @  y  �    7  C  @  0    �  O  �  �  �  \  �  {  �  !  E  �  �  	  	  	  	  �  �  b  �  h  �  �  �  )  �  �  �  �  �  �  �  �  �  �  �  v  c  O  :    �  �  d    �  	?  	�  	�  
  
  	�  	�  	�  	}  	H  	  �  �  �  1  �  �  �  �  N  ;  !  �  �  �  �  j  A    �  �  �  h  ;  �  �  l  �  �  �  �  �  �  �  �  �  �  �  w  Z  ;  F  K  ;  +      �  �  t  �  �  �  �  �  �  �  �  �  u  X  3    �  �  q  :  �  N  :  <  ?  A  D  G  I  L  O  R  O  G  ?  7  /    �  �  �  �  	Q  
  
s  
�  
�  
�  
�  
�  
�  
�  
�  
�  
  
  	�  	  A    �  B  	�  
X  
�  
�        
�  
�  
�  
4  	�  	w  	  �  �  /    �  1  �    "  #        �  �  �  `     �  �  2  �  w    �    �  �  �  �  �  �  �  �  {  h  U  A  -      �  �  �  �  �      �  �  �  �  �  �  �  �  ~  m  \  K  :  *    �  �  �  |  �    )  E  a  n  o  ]  4  �  �  >  �  G  �    �  �  �  Z  <       �  �  �  �  ~  c  A    �  �  �  �  �  �  ~  j  �  �  �  �  �  �  �  �  �  �  l  >    �  �  _    �  �  �  �  {  m  `  R  D  4  $      �  �  �  �  �  r  V  9       A  u  �    !  )       �  �  �  N  	  �  O  �  ^  �  �  :  F  0    �  �  �  �  �  v  W  3    �  �  Q    �  �  �  h  �  �  �  �  �  �  �  �  �  �  �  �  u  e  U  F  5  %      �  �  �  �  �  �  �  �  �  �  c  3    �  s    �    U  3  �  x  o  e  U  C  2      �  �  �  �  �  �  �  w  m  c  Y    �  �  �  �  �  ~  Z  /    �  �  |  5  �  �  +  �  6  �  9  5  0  +  %            �  �  �  �  �  �  x  _  I  4    �  �  �  �  �  o  W  <  !    �  �  �  �  k  @    �  m    K  M  P  9      �  �  �  �  s  O  )    �  �  �  `  6    N  @  2  $    	  �  �  �  �  �  �  c  >    �  �  f    �  �  x  p  d  _  y  �  �  �  u  m  j  c  Z  P  E  Z  v  s  l  �  �  �  �  �  �  �  �  q  ]  =    �  �  �  �  a  ?     �  �  z  _  B  +  B    �  �  i  )  �  �  [    �    �  -  �  '         �  �  �  �  r  S  1    �  �  [    �  �    K  
�  ~  �    %  ,    �  �  �  I  
�  
|  	�  	K  �  �  �  �  Q  �      �  u  ]  A  "  �  �  �  U    �  �  `  "  �  �  �  v  g  X  D  3       �  �  �  y  I    �  �  ?  �  �    Q  \  S  J  B  9  .         �  �  �  �  �  �  �  �  �    #  �  �  t  a  N  :  %    �  �  �  �  o  5  �  �  ,  �  1   �  �  �  �  �  �  �  �  �  a  5  �  �  y  !  �  w  �  K  u  �  n  \  I  *  
  �  �  �  �  |  c  O  >  2  /  (    �  �  �  �  �  �  �  ~  l  Z  I  4      �  �  �  �  j  F  !   �   �  �  �  �  �  �  �  n  I    �  �  Z  �  �  8  �  y    �  �  �  �  �  �  �  �  w  ^  C  %    �  �  }  <  �  �    �    �  �  �  }  n  ^  P  B  4  &    �  �  �  �  �  �  �    q  q  ]  H  1    �  �  �  �  �  j  P  4    �  �  �    ?  �  	  �  �  �  _  :    �  �  �  b  )  �  p  �  l  �  8  r  �  �  �  �  �  �  z  F    �  �  �  c  =    �  �  �  �  `  ;  N  ?  1  "      �  �  �  �  �  �  �  Z  -     �  �  ]  &  	�  	�  	�  	�  	U  	  �  �  �  W    �  W  �  �    Q  t  �  �