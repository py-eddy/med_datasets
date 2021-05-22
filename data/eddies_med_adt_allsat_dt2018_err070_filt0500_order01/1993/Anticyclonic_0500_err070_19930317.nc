CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�p��
=q     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��S   max       P#=�     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       =���     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F�Q�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vq\(�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q`           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @�۠         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >C��     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��	   max       B.(n     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B.�     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�f%     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�"   max       C�g�     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          s     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          +     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��S   max       O�"     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Q�   max       ?�H˒:*     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       =�G�     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @F�Q�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vq\(�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q`           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @��         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?�E8�4֢     �  _�                                                 
                        <   &         *      
                  2   %   /         #   (   !      
   3               !      &             A               
   	            '            6   s   O� PCO�N��N��5N��&NUA4Ow}%ND��Ob��O[+�N��oNI�_NټO���O�O�c_O?N,��N\ N�YO�dM��SO�2'O���N�"NOK{�O0�OfW�N��N�D�N���O���N��N��<O�sO��rO�q|Nى�O)�O�O�,O�"OEEN�_�P#=�NѕpN�N���N��;O��{Oi2fO6y�N��O�L�OP�NNNi+;O�1�N��N��zN�yN&"�O+�*O=hO���N���NC��O�Oڏ�O�6�O@�ļD���D���#�
��`B�o��o;o;D��;��
;ě�;ě�;ě�;ě�;�`B;�`B;�`B<t�<t�<t�<#�
<#�
<49X<49X<49X<D��<D��<D��<T��<T��<�o<�C�<�t�<���<��
<�1<�1<�1<�9X<�j<�j<ě�<ě�<���<���<���<���<���<�h<�<�<��<��=o=o=o=+=+=+=+=+=�P=�P='�=,1=P�`=P�`=q��=q��=y�#=�+=�hs=��-=���W^cdz������������zaW-,5B[gt�������zg\J5-�������������������������������������~~����������������adhntv�������tqhaaaawuy�������wwwwwwwwww+5BNSVXYPB5)& ��������������������?>?BEN[]gt|tkg[UOB?����
#/4:3,*'# 
��*,/<<HJPXWUKHF<;95/*��������������������OBB?BGO[^[ZOOOOOOOOO��������������������<=BJN[gsstttjg[NFB<<2,*+14:BN[gruurmg[B2������������������������������ ��������=9BOO[ehnplhe[OB====������!(%��������������������������)6BGMOYTOH6)�		"/8HTjgaTH/��������������������@>>;6ENX[gijong[NJB@Z^fhptz��������thb^Z��������������������bgit���������tigbbbb��������������������#04310'#`gt�������������}nk`MJIKOT[]dhjkkha[WOMM�������������������������������IIORX[ht~�����{thaOIqmnuuw�������������q��������������������,(&&(0<=FMPRRJI<910,��������������������#<HMU`c\JHH</ ���	)5N[aaVB+��)5>BNVVNB5) ))568875)���)1<IEGB5)��������������������������������)5:52)��������������������(%*25NV[_cggfa[NB51(LNQ[gt����ztmga[VNL#$./<@BDEB<8/#yz}�������������|zyy�����*.41,�����������������������������
#<Uqxvl^H</#
��#/250/-#52<?HKSUYUH<55555555EDGKUaouz��{qnaUNHE������������������������ 
"#$#
���#"#0;<HH@<00######&#


����������'  )6BDOY[g[XOCB6,'�����������������������
"&!����

�����������]Yamz{~���{zomjfdca]x}|���������������zx��������

������������������������������������������������������������������������!�3�7�)������ùòùú�������޻������û˻лܻػлû��������������������������������������������}�~�������������m�z�������������z�v�m�e�d�h�m�m�m�m�m�m�����ȼʼ˼ʼ¼��������������������������n�zÇÎÇÇ�z�n�l�b�n�n�n�n�n�n�n�n�n�n�uƁƎƑƚƠƥƤƚƎ�u�h�\�V�U�Y�\�h�p�u�(�5�A�N�T�N�A�=�5�)�(��(�(�(�(�(�(�(�(���)�0�5�A�C�B�;�/��������������
��`�m�y�������������y�m�`�T�;�4�;�G�I�T�`�	�� ����	������׾Ծ׾������	�(�1�4�?�9�4�(� ����#�(�(�(�(�(�(�(�(�l�l�l�y�����������}�y�l�l�l�l�l�l�l�l�l����*�C�O�W�h�Z�O�J�6�������������������������������������������������������������$�0�D�K�D�9�$�������������廷�ûлܻ����������ܻлƻû��������y���������������y�w�p�n�y�y�y�y�y�y�y�yù����������������ùõðùùùùùùùù�<�H�L�T�U�^�a�e�a�U�H�?�<�1�2�8�<�<�<�<�`�m�y�����������������y�m�i�e�a�`�\�`�`���������������������������������������Ҽ�����������������z�r�f�c�Y�F�@�F�X�u���"�/�;�T�a�k�s�u�m�[�R�H�;�5�������"������������������������������������.�;�G�T�`�m�p�m�k�j�`�T�G�;�,� ��"�+�.�L�Y�e�j�r�|�r�o�e�Y�L�@�9�3�.�-�/�3�@�L����'�3�;�A�?�3�'��������������"�*�/�3�4�/�(�"���������"�"�"�"�	��"�&�&�'�#�"� ��	����������	�	�	�	������������߼ּӼּ߼������������������������s�f�Z�N�C�C�M�f�s�������������������������s�j�s�w������������������������������������������������(�M�Z�f�k�t����s�Z�M�A�4�(������(������(�2�@�4�(������ݽн̽ʽн��������4�M�Y�S�J�@�4������ܻлʻһ�T�`�c�m�y�}����z�y�y�m�`�V�T�K�L�T�T�T�����Ľнݽ���ݽнĽ������������������ѿ�����(�,�/�)�������ݿ̿ɿʿ˿���������������������ùò���������
�#�<�U�n�|ŋŎōņ�}�n�b�U�G�<�����
��������������������ļĳĚĒĖĝĨĳĺ���t¦²µ²¦¦�t�r�h�n�t�t�	�"�2�H�O�O�G�;�"���������������������	�/�<�@�H�M�H�H�A�<�<�=�:�/�)�#�!��#�/�/�/�;�H�J�O�M�H�;�/�"�����"�'�/�/�/�/�f�n�q�q�r�f�]�Z�V�V�Z�\�f�f�f�f�f�f�f�f�����������������������������������~�{���������%�(�1�(������ѿƿ����ȿѿ������������������s�g�Z�N�D�A�;�A�K�Z�w��D�D�D�EEEEE*E/E5E*EED�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٽ}�������}�}���������l�`�S�J�J�S�`�l�u�}āčĚĠĩĩĦĚĒčā�t�q�t�u�t�s�t�zā��(�8�8�4�8�8�/�������ڿֿؿ߿���DVDbDkDfDbDVDIDIDIDMDVDVDVDVDVDVDVDVDVDV���	�����	����������������������������������������������������������������������������������������������������)�5�<�B�N�O�N�K�B�5�/�)�'�������Y�f�r�������������r�f�c�Y�Y�Y�Y�Y�Y�Y�ʾ����������������ɾʾʾʾʾʾʾʾʾʾʻF�S�_�l�w�x�y�x�l�g�_�S�F�:�7�+�)�:�@�F�-�:�F�S�V�U�S�S�M�F�B�:�3�-�(�!� ��!�-����'�3�@�F�[�Y�L�@�3������ٹٹ��e�k�r�~���������������~�w�r�n�e�c�\�e�eǮǴǭǪǡǘǔǓǔǚǡǮǮǮǮǮǮǮǮǮ����������������ŹŭŠŞŠšŭŹ�������߼����ʼּڼ��������ʼ��������q�s������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDxD{DD�����
�#�/�<�H�O�P�H�<�#��
������������ . 1 3 - T H 4  Z ! ? l . Q P 1 + s @ \ X 5 � , * H /  : W 3 ' B n \ 9 0 B ; R % / r ~ X ^ d / f m 8 * H  k 4 W ? 2 1 M 4 , I ' A G z D p h   p      �  #  �  �  �  _  �  Y  �  �    S  2  �  @  �  �  [  \    S  j  "    �  �  r  �    �  �  a  �  �  �  5  �    �  �  T    	  ,  e  ,  �  �  1  l  �  �    �  %  �    t    "  �  �  @  g  e  �  �  K  d  B    �<#�
<���<t��o;o;��
;�`B<ě�<49X<���<�j<D��<49X<u=+<�C�=+<�j<49X<�9X<�9X<�1<�t�=�\)=H�9<�j=+=]/<�h<���<ě�<�j=o<ě�<�`B=�\)=ix�=�C�<�`B=P�`=m�h=�o=m�h=H�9=\)=��-='�=�P=\)=#�
=�o=<j=�\)=�%=8Q�=L��=Ƨ�=49X=#�
=}�=H�9=<j=H�9=<j=��-=��T=Ƨ�=�o=�C�=��->   >C��=�v�B �.B	-	B#N�B!<B.Bt�B
b�B��B��B�0B�JB��B"�\B��B�jB��BT B�>B*�B�B�B{XBPB	�A��	B!4By{Be�B!g+B	�JB�3B%IABG�BCB#B�9B�lBB �zB&:�BT�B��B�BE�B�BDB I�BٻB^�B**BSB	]:B1-B%~B.(nB��B�B�B�B=�Bi�B�B%��B$��B��B�B	�B�B��A�v�B\�BނB��B @�B	<6B#@�B!?�B?*BI1B
~`B�DB��B�7B0B?0B"�~B�~B��B�hB�'B�BͩB?�BE�B��B�hB�\A���B>�B��BMB!�B	ζB��B%?BO�B��B�B��B��B<]B ��B&A�B{jB��B=>BE�B�ZB?LB C�B4�BF�B*CEBAB	�7B>�B>�B.�B�B�B/;B�B@�B@B�5B%�=B$¦BJ�B?sB��BNcB��A�C�B�2BİB��A��0A�[�@���A��A���@���A�U(B��A��A�OqAjE�AXcA7=�A�}A�w�A�B�@�z�Am��A�q�AĠLAl��A�7@��A�x�B��AdU�?ˋz?� �A�Y�A�l�A�RAB>sAH�A��A<AvA0�1@�#jAi�nA&�]A��&A�aA���AἙA���A�XA���A���AA$*ATpA��A��C�QC�f%Ad�A�W�A�,�C�|�A��A�9A��0A�v@�rAM��@�*?@|�=?��@NWB��A���@�3�C���A���A���A�i�@���A_A��T@�@�AȈXB�@A�i�A���Aj�AV0A7ܡA^A�YuA���B�@�#�Am�A΀A�o�Am�A�{�@�A���B��Ac g?�� ?���A�|A�Z�At�ACeAJ�A���A<p&A/�@�zAh�A&��A�$A�l@A�ruAᩞA�`�A��A~A�z�A@�A �MA�A���C�H�C�g�A�2A�lA��C�xFA�|OA��A�%A��:@��#AM&a@�A@|#?�"@65B�)A�GA �kC��kA�Q                                           	                              <   &         *      
                  2   %   /         #   )   "         4      	         "      '             A               
   	            '            7   s      !   )                                       !                           '   '                                 !      !               +         +                                 )                              !            )            #                                                                                                                           +                                                                        !            )      O��1O��sN�hN��N��5N��&NUA4O��ND��Ob��N�� N��oNI�_NټOr_N�VO��O?N,��N\ NT�O�dM��SO�L�O��N�"NOG�N�6)N�N~<_N�D�NO][O6UhN��N��<O]�vOr��O�+bN�WXN��O�F�N���O�"O`�N�_�O���N��N���N���NC��O{�Oi2fO��N��O�L�N���O}��NNNi+;O^�uN��N�C�N�yN&"�O�N��O���N���NC��O�Oڏ�O��O�m  Q  o  �  �  >  l  v  #  e  P  f    6  �  �  �  �  �  ^    _  <  �  #  �    Q  U  �  	  �  �  B  f  7  Y  �  �  �  �  m  �  �  �  �  C  |  H  f  �  �  �  ^  �    	  �  f  �  g  I  L    �  �  �  �  �  �  t  �    9�t��o�ě���`B�o��o;o<t�;��
;ě�<T��;ě�;ě�;�`B<T��<o<D��<t�<t�<#�
<T��<49X<49X<�h<�t�<D��<u<ě�<��
<�C�<�C�<���<�9X<��
<�1=\)<���=o<ě�<��<�h=,1<���<�<���=<j<�`B<�<�=+=t�<��=��=o=o=C�=m�h=+=+=�P=�P=��='�=,1=aG�=q��=q��=q��=y�#=�+=�hs=�G�=��T^]aikz������������q^105D[gt�������tg[Q51���������������������������������������~~����������������adhntv�������tqhaaaawuy�������wwwwwwwwww&"$)+5BFNQSSRNB?53)&��������������������?>?BEN[]gt|tkg[UOB?���
"##$##
 �����*,/<<HJPXWUKHF<;95/*��������������������OBB?BGO[^[ZOOOOOOOOO��������������������>?BLN[gprrhg[NIB>>>>1./48BN[gorspjg[B<51������������������������������ ��������FCOR[hmhf\[OFFFFFFFF������!(%����������������������� )6<?FE=6) "/;HTZ^RH;/"��������������������@A?BENW[fgllgf[NNCB@dachlt��������vtrhdd��������������������ggmt�������tkggggggg��������������������#01100%# z{�����������������zMJIKOT[]dhjkkha[WOMM�������������������������������KLRU[hkty������thfPKrsz{y{�����������wr��������������������-)*-0<AHILMKI<10----��������������������-#%$//6<HITOHA</----���	)5N[aaVB+��)5;BENSRNB@5/)#))568875)�� ).46563)
��������������������������������)5:52)��������������������4+).5BIN[addca^[NB54LNQ[gt����ztmga[VNL #)/<=?AB?<3/#  yz}�������������|zyy�����*.41,��������������������������#<HUZade\UH</##/250/-#52<?HKSUYUH<55555555GFHLQUadnpz~|wnaUPKG�����������������������	
!!
�������#"#0;<HH@<00######&#


�����������)()6BOPXPOB6.)))))))�����������������������
"&!����

�����������]Yamz{~���{zomjfdca]x}|���������������zx�������


�����������������������������������������������������������������������	���-�3�)��������ÿ�����������޻������ûĻлӻллû��������������������������������������������}�~�������������m�z�������������z�v�m�e�d�h�m�m�m�m�m�m�����ȼʼ˼ʼ¼��������������������������n�zÇÎÇÇ�z�n�l�b�n�n�n�n�n�n�n�n�n�n�h�uƁƊƎƖƛƚƗƎƁ�u�h�d�\�\�[�\�c�h�(�5�A�N�T�N�A�=�5�)�(��(�(�(�(�(�(�(�(���)�0�5�A�C�B�;�/��������������
��m�y�z���y�m�l�`�T�T�O�Q�T�`�d�m�m�m�m�	�� ����	������׾Ծ׾������	�(�1�4�?�9�4�(� ����#�(�(�(�(�(�(�(�(�l�l�l�y�����������}�y�l�l�l�l�l�l�l�l�l���*�;�C�O�O�K�C�6�*������������������������������������������������������������$�:�?�?�3�$�����������������򻷻ûлܻ����������ܻлƻû��������y���������������y�w�p�n�y�y�y�y�y�y�y�yù����������������ùõðùùùùùùùù�<�H�P�U�\�^�U�K�H�<�;�7�<�<�<�<�<�<�<�<�`�m�y�����������������y�m�i�e�a�`�\�`�`���������������������������������������Ҽ������������������r�f�Y�Q�J�Q�Y�d�r������"�;�H�T�a�f�o�p�d�T�K�H�;�"�����������������������������������������;�G�T�\�`�h�`�^�T�S�G�;�/�.�"�"�"�.�2�;�@�L�Y�`�e�m�i�e�\�Y�L�G�@�7�3�3�3�8�@�@��'�3�6�7�;�5�3�,�'����	�������"�#�/�1�2�/�#�"� �����!�"�"�"�"�"�"�	��"�&�&�'�#�"� ��	����������	�	�	�	��������������ּռּ��������f�s������������s�f�d�Z�W�M�K�J�M�Z�f������������������������s�j�s�w������������������������������������������������4�A�M�Z�b�l�s�t�s�m�f�Z�M�A�4�'���(�4������'�7�4�.�(�������ݽнνн�������'�4�@�M�O�K�@�4�'������ۻܻ���`�a�m�w�y��z�y�m�`�Z�T�T�O�T�]�`�`�`�`���Ľнݽ��ݽнϽĽ��������������������ݿ�����*�,�(�$��������ѿοѿտ�����������������������������������
�#�<�U�n�|ŋŎōņ�}�n�b�U�G�<�����
Ŀ������������ĿĸĳĚĖęĚĠĦīĳľĿ�t¦²µ²¦¦�t�r�h�n�t�t���	��"�/�7�9�+�"�	���������������������/�<�H�K�H�F�>�<�8�/�*�$�#�!�#�%�/�/�/�/�/�;�H�H�L�H�H�;�/�"���"�+�/�/�/�/�/�/�f�n�q�q�r�f�]�Z�V�V�Z�\�f�f�f�f�f�f�f�f�����������������������������������������ѿ��������������ݿѿ˿ÿÿο������������������s�g�Z�N�D�A�;�A�K�Z�w��D�D�D�EEEEE'E"EEED�D�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٽ}�������}�}���������l�`�S�J�J�S�`�l�u�}āčĚĞħĨĦĚďčā�t�r�t�v�t�t�tāā�����%�)�&���������������DVDbDkDfDbDVDIDIDIDMDVDVDVDVDVDVDVDVDVDV���	�����	���������������������������������������������������������������������������������������������������)�5�9�B�N�N�N�I�B�5�)�!� �"�)�)�)�)�)�)�Y�f�r�������������r�f�c�Y�Y�Y�Y�Y�Y�Y�ʾ����������������ɾʾʾʾʾʾʾʾʾʾʻS�_�c�l�u�v�l�`�_�S�F�E�:�2�0�:�F�I�S�S�:�F�H�Q�O�G�F�:�-�-�&�!�-�9�:�:�:�:�:�:����'�3�@�F�[�Y�L�@�3������ٹٹ��e�k�r�~���������������~�w�r�n�e�c�\�e�eǮǴǭǪǡǘǔǓǔǚǡǮǮǮǮǮǮǮǮǮ����������������ŹŭŠŞŠšŭŹ�������߼����ʼּڼ��������ʼ��������q�s������D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD~D�D�D�D����
��#�/�<�H�L�N�H�D�<�/�#���������� . 3 7 - T H 4  Z ! , l . Q A 3 % s @ \ _ 5 � - $ H *  G - 3 + ; n \ 2 1 I ? A " $ r e X N \ 4 f P , * 8  k 2 > ? 2 5 M + , I ) 3 G z D p h  a    �  %  �  �  �  �  _  C  Y  �  �    S  2      6  �  [  \  �  S  j    �  �  4  �  �  �  �  ]  �  �  �  �  �    �  �  D  �    D  ,  "  �  �  �  r  �  �  :    �        t  �  "  �  �  @    �  �  �  K  d  B  R  t  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  H  O  Q  O  K  L  N  I  A  7  $    �  �  �  ~  8  �  C   �  L  d  n  l  ^  D     �  �  �  j  N  @  :  4  �  �  �  �    �  �  �  �  �  �  �  �  |  a  B    �  �  W  �  X  �  �  =  �  �  �  �  �  �  �  �  �  �  �  ~  }    �  �  �  �  �  �  >  3  )      
  �  �  �  �  �  �  �  �  ~  Z  /     �   �  l  g  c  ^  [  X  U  P  J  D  =  3  *      �  �  �  �  �  v  c  P  =  )       �  �  �  �  �  �  u  e  T  D  5  %    �  �  
      #  !      �  �  �  �  y  B     �  k  2  �  e  U  E  6      �  �  �  �  �  g  J  +     �   �   �      ^  P  E  9  +    
  �  �  �  �  �  �  �  t  U  0    �  n   �  8  Q  Z  ^  \  T  I  K  e  `  S  <    �  �  �  r  S  P  Y      �  �  �  �    �  �  �  �  �  �  �  �  �  g  Q  :  #  6  -  #        �  �  �  �  �  �  �  �  �  �  {  n  a  T  �  r  X  J  C  ;  .  !    �  �  �  �  �  r  V  9    �  �  E  Z  m  |  �  �  }  q  `  L  8      �  �  x  %  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  z  W  /  �  m     �  �  �  �  �  �  �  �  �  �  }  p  a  L  0    �  �  ]     �  �  �  �  �  r  q  Z  /    �  �    i  ;    �  y    �  Q  ^  U  L  C  :  1  (                �   �   �   �   �   �   �      �  �  �  �  ^  -  �  �  f    �  �  >  �  �  N   �   �  �  
  (  A  S  a  m  t  y  w  u  q  k  }  �  �  �  �  �  �  <  2  '    
  �  �  �  �  �  �  t  ]  E  +    �  �  �  [  �  �  t  i  k  l  u  �  �  �  g  M  2    �  �  �  �  �  f  /  �  �  �      #    �  �  �  J  �  �  7  �    {  �  )  �  �  �  �  �  �  �  �    ^  :    �  �  |  4  �  P  �  �      �  �  �  �  �  �  �  o  U  =  $    �  �  �  �  �  v  	  #  F  Q  N  H  H  J  M  G  :     �  �  �  +  �  c  �  q  s  �    0  H  T  Q  G  /    �  �  �  V    �  �  B  �  ~  J  H  Q  ]  i  u  ~  ~  �  �  �  �  �  b  5    �  �  �  1  �  �        �  �  �  �  �  �  �  �  �  �  �  t  [  :    �  �  �  �  �  �  �  �  �  �  t  ^  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  `  K  4       �  =  :  9  9  @  =  3  '              �  �  �  �  n  9  f  \  S  J  A  8  .  $      �  �  �  �  �  �  �  t  f  W  7  4  1  ,  $        !        �  �  �  �  �  �  �  �  �       4  J  W  X  M  ;    �  �  I  �  �    �    �  �  h  z  �  �  m  N  )  �  �  �  t  M  (    �  �  �  @  �  �  �  �  �  �  �  �  �  �  �  k  4  �  �    �    �  s    �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Z  G  3        �  Q  �  �  �  �  �  �  �  �  �  ~  J    �  �     �  #  �  �  N  `  k  l  j  d  W  C  %     �  �  Y    �  ?  �  K  �    �    @  �  
  ?  g  �  �  �  d  :    �  �  E  �  H  �    �  {  c  A    �  �  �  �  �  �  �  n  5  	  �  `  �  H  �  �  �  �    �  �  �  ~  b  N  #  �  �  H  �  �  $  �  �  *  �  �  p  _  O  F  ;  +      �  �  �  �  �  �  �  �  x  p    r  �  �      9  B  7    �  �  �  d  �  <  ;    �  h  &  f  z  {  u  f  R  9       �  �  �  �  f  K  /  �  �  :  >  B  F  H  G  D  >  8  2  ,  &         	    ,  I  p  �  f  ^  U  M  E  =  5  -  '  #                      w  {  �  �  �  �  �  �  �  �  �    o  _  L  7  !    �  �  j  �  �  �  �  w  [  8    �  �  f    �  S  �      �  �  �  �  �  �  �  �  �  x  e  R  ;    �  �  �  p  5  �  �  �  M  R  Z  ^  Y  I  6    �  �  �  S  	  �  \  �  [  �  �  �  �  r  a  S  <     
  �  �  �  j  '  �  s  �  x  �  -  t  �    �  �  �  �  �  �  �  �  �  �  �  h  F    �  �  \  
   �  �    �  �  �  �  �  �  ~  d  I  -    �  �  �  �  �  ~  q  R  }  �  �  �  �  �  �  �  �  �  �  v  2  �  .  �  �      f  S  A  /      �  �  �  �  �  �  l  M  -    �  �  P    �  �  �  �  �  �  �  �  �  �  �  �  z  u  s  u  x  �  �  �  G  Y  e  f  \  N  @  0  &      �  �  r    �  C  �  A  �  I  B  9  +      �  �  �  �  ]    �  {  &  �  �  �  �  �  <  D  L  D  9  '    �  �  �  �  �  �  �  p  ^  J  2  �  �    �  �  �  �  �  �  �  �  �  p  \  G  2      �  �  �  �  �  �  �  �  �  �  �  �    u  j  ]  P  B  5  !  
   �   �   �  �  �  �  �  �  �  �  �  �  |  c  L  ,    �  �  z  G  �  �  �  2  q  �  �  �  �  �  �  l  L  )  �  �  u  (  �  X  �  	  �  �  �  x  c  S  >  %    �  �  �  6  �  �    �      ,  �  �  �  �  �  y  c  N  7       �  �  �  �  �  T  "  �  �  �  �  r  Y  6    �  �  �  �  f  A    �  �  �    W  0    t  [  A  "    �  �    �  �  �  �  �  s  d  a  e  q  �  �  �  �  �  �  �  p  b  R  >  /    �  �  M  �  n  �  �  �  -    o  �  �  �        �  �  <  �  :  t  �  u      	�  W         8  4      �  �  �  U    �  �  ;  �  �  A  �  P