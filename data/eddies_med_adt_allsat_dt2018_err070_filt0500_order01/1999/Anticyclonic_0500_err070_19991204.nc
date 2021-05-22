CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�`A�7K�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M⨱   max       Q-��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       =�^5      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @E��
=p�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vI\(�     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       A �`          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >�1      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��O   max       B,}%      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?3WP   max       C�5�      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?.P�   max       C�3I      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         N      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          e      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          K      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M⨱   max       P��      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��e+��   max       ?��MjP      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       >�      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E��
=p�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vI\(�     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G:   max         G:      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q��   max       ?��hr�!     �  T         	           N   $   j      	   W      "      /      (         	      5      D            V               )               !            X      .   ?      *         1   
         	               N2{N�6NVTN�
M⨱N0��P�ZO���Q-��N!��N��KQSTN:��O9ƮO�wO��=N�6O�c�O�[�Osr�O��O �.O�dQN�:P�O�+QN��OLP7�-N{��O��LO$�$N"EvOp~@O�O���N�[N�/O�6�NL}_N*kNO+��PC^�O���O\J{P6�"O�1�O�e O��nOYZ[O��OxOb N�N�bO؉N~�.N]�O���N����1�t��o��`B%@  ;o;D��;D��;ě�;ě�;ě�;�`B;�`B;�`B;�`B<t�<#�
<T��<u<u<�o<�t�<��
<��
<�1<�1<���<�`B<�`B<�`B<�h<�h=o=+=C�=C�=\)=\)=\)=��=#�
=,1=,1=,1=0 �=49X=<j=D��=P�`=T��=Y�=Y�=Y�=m�h=q��=y�#=�t�=��
=�^5=�^5����������������������������������������=?@BMO[[WOJB========����������������������������������������ljgnz{~�znllllllllll)N[g����g[B5)����� "�������������NXt�g5��������������������������������������������KHMm��)�����zmfWK�����������������������������������������������������!$/<HU^anpqjbRH/!+/<=HRUPH<3/++++++++O[ht�����������th_OO��������

	�����)06IU\^[UIF<0,#����)1.)"�����������������������{z|������������������������������������YZefchz���������zmaYA==CGHUan{�����znaUALOU[_htzytrh[PSOLLLL)5BNW[XNB5)!�����
/?JOOOH:/����������

������::>Mcht��������thOB:�������������������������������������������	"-1133/" �������!)))(���)$),3@HO[e����t[OB7)�������������������������������!!)4N[gty{{ztg[XE5,!(/1<HIMHH<1/((((((((���������������������������������������������$()#�����SOQ[g�����������tigS�����������������������
#278*
���#)69BBOY[h[MF6)����������������������������������������������	����vvurrwz�����������zvpost�������������vtp����������������������������������������������������������������������������������������������ntz}�������{zwnnnnnn����),0.)����������������������������{ŇŏŔŖŔŇ�{�u�t�{�{�{�{�{�{�{�{�{�{������
�����������������ìù��������ùìàÞàåìììììììì�ʾ׾��������׾־ʾ������ɾʾʾʾ����������������������������������������Ҿ׾�������׾;Ծ׾׾׾׾׾׾׾׾׾����)�6�=�J�M�K�6�����óãäðû����������������ɻĻ��������v�l�_�K�<�F�S�l�����C�uƳ��������Ƴ�C����#����������s������y�w�s�h�k�o�q�s�s�s�s�s�s�s�s�������������ּҼмּ�������������������'�#�	���������,�$�-�>�����˼���������������������������������������Óàìù��������ùîìàÓÇ�~�u�x�zÇÓ���(�4�8�M�O�Z�d�Z�M�A�4�(�����	������������� �����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�\�h�m�j�Y�L�A�4�(������������M�\�N�Z�g�s�������������s�g�Z�N�A�<�9�<�A�N������������������������y�r�c�X�Y�d�r���(�5�A�N�\�a�c�Z�N�G�A�<�5�1�,�(���àìðñïìçÖÇ�z�n�m�s�z��{ÂÇÓàÓàìù��������������ùàÇ�z�o�nÀÇÓ�û˻лԻлʻû��������������������������n�{Ŕŭ��������������ŹŠŇ�w�u�o�p�i�n�����������	��	����������������������˺Y�a�Y�T�Y�_�`�Y�L�B�@�7�7�@�L�R�Y�Y�Y�Y�(�3�4�:�<�:�7�4�(��������(�(�(�(�"�.�T�m�������������y�`�;�-�"��
�
��"���ʾ׾����������׾оʾ����������������4�M�V�X�P�@�'� ���������������'�(�,�(������������������������������z�������������������������a�f�h�k�m�i�a�T�H�/�"�����/�H�O�T�a�/�<�E�H�U�V�U�K�H�>�<�5�/�$�#��!�#�$�/���
���#�/�<�C�B�<�.�/���������������ݿ�����ݿѿĿ����Ŀѿؿݿݿݿݿݿݿ�����������������������������������������(�5�A�P�T�L�<�(����տԿؿݿ����²½¿����¿²¦¦£¦§²²²²²²²²����������������������������������������čĚĦĳĺĿ��ĿĺĿĳĚĖčąĂĀāćč����0�I�Y�d�c�Y�I�0�����������Ŀ�������m�y���������������y�m�T�M�E�D�G�H�T�a�mDoD{D�D�D�D�D�D�D�D�D�D�D�D{DkDcDbDXDbDo����)�B�[�j�p�s�p�g�V�5�)������������������������Ǻɺֺغպɺź������~�q�r�~���f�r������������������r�f�M�H�C�B�L�Y�f�(�4�M�^�d�^�U�P�A�7�(���������(���	��"�;�G�W�`�T�M�G�;�.�"��
���������ûлܻ��������������ܻл������������!�-�:�;�=�:�2�2�-�*�!��
����
���!�����Ľнݽ�ݽؽнĽ������������������������������������������������������������	����"�"�2�3�0�/�"����	�������	�������	�
���
������������������������x���������������������x�v�v�x�x�x�x�x�x�׼����������ּӼʼɼʼ׼׼׼׼׼��������������������������������������������'�(�,�'� ������������� 6 X 0 Q ? [ # ? J � ` R \ 7 Z @ < = N H � q - [ G 2 O W  { C  G S # V U 4 = B c 0 + , =  F / = Y 4 N + T w E j S ] �  Q  -  c  �    >  8  �  
�  �  �  	s  �  �  h  _  �  M  !    �  �  �    �  !  �  i  
  �  �  Z  +  0  1  �  �  �  $  s  W  r  O  /  �    *  |  �  �  p  Y  �  Q  ]  W  �  �  T  ���t���`B%@  ��o;��
<o>�1=�w=�S�<D��<u=�v�<e`B='�<�t�=aG�<�o=T��=49X=C�<���=t�=�t�=\)=�9X=0 �=t�=\)=�l�=\)=q��=0 �=t�=��P=H�9=u=�w='�=�C�=T��=49X=e`B>�=�\)=�E�=�"�=�hs=�Q�=��w=�+=��=}�=��P=�%=�7L=�hs=���=� �=�l�=Ƨ�B�5B�oB��B9�B�BB~B�UB�B�vB��B\B �B"
DB#b�B��B�8B�IB��B&�B�BwB�LB"��B EhBv\B�JBU*B�uB! B�AB!7�B �6A��OB��BOVBوB�B�[B��B"2BH�B�_B
l�B�YBL�B�Bs�BNB?�B�ABE�B+p�B̎BB�B,}%B��Bb_B��B��Bo\B�B6�B�*B=�BD�B��B�uB��B��B�rB <�B"@rB#{B٥B�B�GB��B%̯B�UB:B��B"��B >�BRwB�ZBs`B�DB>�B��B!|�B ��A���B��BAQB�rBaB�$B��B",bB��B�`B
 �B��B?�BWB?�B9�B@�BoLB�;B+C8B�eBTzB�B,��BFB4�B��A�$?3WPA��AS��A�2|AU&�A��z@�B~AB��A�A���@�+mAˣ�A7�:A�e�C�5�A8C:A�
j@�b�A�N�A�P#A��@��A���A�U�?�8|A7 �Ai��AT�@Ū�@���A���A�ٮA�?A�FHA{��At�\A��1A��JA!AA���A鏶Am��C���A��@4|@���A9�A_�@�!@kzGA#�WA���A��=A��@��9AAA�Ω?���A�j?.P�A���AT�A�b�AS�A�w�@���B:xAC�A(A���@��A�j�A6S�A��C�3IA8�A�r�@�%IA�}qA��A̅:@�	"A��A��?ǘ+A7FAi��AQ�@���@�`A�A��wAÅA��yA|?SAt+�A�{�A���A ��Aߥ�A��Al��C��,A��@&@޳�A9;TA_�@��@c��A"�cA�A��hA��@���A^<A��?���         	           N   %   j      
   W      #      /      (         
      5      D            V               *               "            X      /   @      +         1   
         	                                    5   #   e         O                  '                     '            '      '               #         #            -   #      )                                                               !      )         K                  !                                                               #            %         %                                          N2{N�6N5�1NHPM⨱N0��O�#�N��OPdiN!��N��KP��N:��Oh�N�Y�N�T�N�6O��$O�[�Osr�O��O �.O�]�N���Ov~�O�sNn#vOLO=YeN{��ON{�O$�$N"EvO>O�Oj+�N�[N�/O��cN<JN*kNO+��PeMO�6�O��P�ZO�1�O��O��nOYZ[O)�jOxOb N�N�bO؉N~�.N]�O9ޔN��      Q  �    �    a  
  �    �  �  S  �  9  U  	  �  [  K  �  	%  Y  	w  �  R  �  	�  h  K  V  S    :  �  K  �  [  p  �  z  	�    �  n  Y    a  "     �  a    q  6  �    �  T��1�t���`B�ě�%@  ;o>�<��
=q��;ě�;ě�<D��;�`B<T��<t�<�<#�
<�1<u<u<�o<�t�=+<�1=<j<�`B<���<�`B=���<�`B=#�
<�h=o=�w=C�=�w=\)=\)=t�=�w=#�
=,1=u=H�9=e`B=aG�=<j=H�9=P�`=T��=�\)=Y�=Y�=m�h=q��=y�#=�t�=��
=ě�=�^5����������������������������������������>?ABIOYYROMB>>>>>>>>����������������������������������������ljgnz{~�znllllllllll)).5BN[gr~�{tg[NB5+)��������������#)/1-&������������������������������������������NMZ����&�����ojiN������������������������������������������������  ������-&(+/6<BHUUWUQH@</--+/<=HRUPH<3/++++++++VX[ht���������tqhd]V��������

	�����)06IU\^[UIF<0,#����)1.)"��������������������������������������������������������������mosz����������zvqoomTMKRUYanqz}��zunaUTTRU[`htyytqh[RRRRRRRR)5BNW[XNB5)!
 #/0<<??<5/#
�������

������AAFKT[cht���th_[VOBA����������������������������������������	��	"*.//11/&"!	����!)))(���24;HO[hw����}t[OB862�������������������������������#"*6BN[gty{{ytg[F5-#)/2<HIMHE<4/))))))))����������������������������������������������#$������e[\agt����������toge���������� �������������
#,/- 
�����#)69BBOY[h[MF6)����������������������������������������������	����z{�����������������zpost�������������vtp����������������������������������������������������������������������������������������������ntz}�������{zwnnnnnn�����&+)"�������������������������{ŇŏŔŖŔŇ�{�u�t�{�{�{�{�{�{�{�{�{�{������
�����������������ìù��������ùìàßàæìììììììì�׾ݾ��������׾ʾ��ʾξ׾׾׾׾׾����������������������������������������Ҿ׾�������׾;Ծ׾׾׾׾׾׾׾׾׾�����$�-�0�0�)����������������������_�l�x�����������������x�l�h�_�S�O�S�W�_Ƨ��������������������ƳƚƁ�u�\�T�aƊƧ�s������y�w�s�h�k�o�q�s�s�s�s�s�s�s�s�������������ּҼмּ��������������������%��	�������=�.�-�A�Z���˼���������������������������������������àìù��������ùìèàÓÇÅ�|ÀÇÓÜà��(�2�4�A�H�M�W�M�A�4�(����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�M�Z�]�d�c�\�S�A�4�(������������4�M�N�Z�g�s�������������s�g�Z�N�A�<�9�<�A�N������������������������y�r�c�X�Y�d�r���(�5�A�N�\�a�c�Z�N�G�A�<�5�1�,�(���àìðñïìçÖÇ�z�n�m�s�z��{ÂÇÓààìù��������������ùìàÓÇ�{�zÌÓà�������������ûлѻлɻû�������������������������������ŹŭşŔŏŋōŔŠŭŹ����������������������������������������˺@�L�R�Y�^�_�Y�L�C�@�8�9�@�@�@�@�@�@�@�@�(�3�4�:�<�:�7�4�(��������(�(�(�(�`�m�y�����������z�y�m�`�T�G�@�?�F�G�T�`���ʾ׾����������׾оʾ����������������'�4�<�@�F�L�@�4�%���������������'�(�,�(������������������������������z�������������������������;�H�T�a�h�j�f�a�T�N�H�/�"�����!�/�;�/�<�E�H�U�V�U�K�H�>�<�5�/�$�#��!�#�$�/�
��#�/�9�<�6�/�(�#���
�����������
�ݿ�����ݿѿĿ����Ŀѿؿݿݿݿݿݿݿ�����������������������������������������(�5�A�G�P�S�K�:�(����޿ֿۿ��	��²¼¿��¿¿²¨¦¤¦¨²²²²²²²²����������������������������������������čĚĦĳĺĿ��ĿĺĿĳĚĖčąĂĀāćč�
��0�<�N�U�X�V�J�0������������������
�m�y���������������y�m�c�T�N�L�P�U�`�k�mDoD{D�D�D�D�D�D�D�D�D�D�D�D{DuDoDlDiDoDo��)�5�N�\�e�j�f�^�J�5��������������������������Ǻɺֺغպɺź������~�q�r�~���r�����������������r�f�M�I�C�C�L�Y�f�r�(�4�M�^�d�^�U�P�A�7�(���������(���	��"�;�G�W�`�T�M�G�;�.�"��
�������ܻ���������������ܻ׻лĻû��ûŻлܻ!�-�:�;�=�:�2�2�-�*�!��
����
���!�����Ľнݽ�ݽؽнĽ������������������������������������������������������������	����"�"�2�3�0�/�"����	�������	�������	�
���
������������������������x���������������������x�v�v�x�x�x�x�x�x�׼����������ּӼʼɼʼ׼׼׼׼׼����������������������������������������Ѻ��'�(�,�'� ������������� 6 X 1 K ? [  E ( � ` P \ 9 K & < < N H � q ) Z 9  R W  { r  G W # J U 4 9 B c 0 # / 8  F / = Y 1 N + T w E j S V �  Q  -  M  b    >  �  4  �  �  �  	B  �  '       �  �  !    �  �  �  �  �  /  �  i  �  �     Z  +  �  1    �  �    b  W  r  S  X  ,  p  *  R  �  �  w  Y  �  Q  ]  W  �  �  �  �  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:  G:        
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  y       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  k  _  B  I  O  Q  O  N  M  L  @  2  #    �  �  �  �  s  I    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  z  o  e  [  R  I  ?  �  �  �  �  �  �  �  �  �      !  )  1  9  A  H  O  V  ]  �  �  Q  d  3  �  L  �  �  �  �  U  �  �  [  �  �    �    }  �  �      #  /  @  V  `  Q  6    �  Y    �  q  >  �    �  �  �  �      >  �  �    �  �  y  9  �  �  �  �  �  �  �  u  i  [  N  @  %    �  �  �  �  �  �  �  �    �      �  �  �  �  �      �  �  �  �  �  �  t  S  ,     �  E  �  �  �  �  �  2  �  W  �  �  >  �  k    �  �  �  �  �  �  �  �  �  �  w  m  c  Z  P  K  H  E  K  Q  Y  c  l  w  �  �  �  !  A  R  Q  C  3       �  �  �  W    �  @  �  �  5  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  k  Z  K  ?  3  '  �  �  �  W  �  �    )  6  5    �  �  G  �  n  �  9  �  �  U  H  ;  .      �  �  �  �  �  �  �    n  ^  M  =  .    �  �  �    	       �  �  �  �  �  U    �  _  �  �    W  �  h  O  1    �  �  �  p  >    �  b    �  e     �  �  .  [  W  O  B  -    �  �  �  �  �  �  �  �  �  �  �  _  �  f  K  +  
        	  �  �  �  �  �  �  �  _  ,    �  �  �  �  �  �  �  �  h  b  S  <  #    �  g  �  �  F  �  �  F  �  v  �  �  	  	!  	"  	  �  �  �  Y    �  e  �  T  �  �  �  �  J  V  G  -    �  �  �    �  �  �  q  *  �  �  .  �  v    h  �  `  �  �  	4  	e  	v  	q  	Z  	.  �  �  Q  �  f  �  t  �  �  j  {  �  �  �  �  �  �  �  �  �  {  \  7    �  �  Y    �  �    R  R  P  N  K  H  F  C  >  3  &    �  �  \  �  s   �  �  �  �  �    y  s  k  d  Z  P  F  :  .  #      �  �  W    X  �  .  �  �  	=  	y  	�  	�  	�  	�  	�  	7  �  "  }  �  J  "  h  c  ^  V  H  :  &    �  �  �  �  g  C    �  �  �  :   �  �  �        '  F  J  >  +    �  �  �  �  v    �  �  t  V  R  H  :  +      �  �  �  �  w  f  S  >  #  �  �  �  �  S  K  C  ;  3  +  #      
  �  �  �  �  �  �  �  �  n  ]  
�  
�  
�    
�  
�  
�  
}  
8  	�  	�  	  {  �  Q  �  6  �  �  �  :  2  )      �  �  �  �  �  �  �  h  L  /    �  �  i  #  �  �  �  �  �  �  �  �  �  �  l  A    �  �  d    �  ,   �  K  C  ;  3  +       �  �  �  �  �  z  X  6    �  �  �  \  �  �  �  �  �  �  �  �  �  �  �  �  z  j  [  L  =  /  !    Y  U  D  1       �  �  �  �  |  X  4    �  �  k    +  �  c  m  u  r  W  O  4    �  �  w  =  �  �  %  �  B  �  P  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  o  k  f  b  ]  z  p  c  S  @  4    �  �  �  �  j  P  8       �  �  G  �  �  	  	[  	�  	�  	  	j  	P  	C  	:  	#  �  �  �  $  �    %  �  �      	        
    �  �  �  �  �  e  B       �  �  0  �  �  �  �  �  �  �  �  �  �  �  ^    �  �           �    <  Z  l  k  [  D  "  �  �  n    �  S  �  K  �    G   �  Y  P  B  3  $      �  �  �  �  X  *  �  �  H  �  �  -  A  p  |  l  W  ?  !  �  �  �  c  0    �  �  [  �  �  �    �  a  C  $    �  �  �  �  �  �  �  �  �  �  X  '  �  �  �  a  "             �  �  �  �  �  a  3  �  �  �  f  &   �   �  �  �  �  �            �  �  p     �  a  �  Y  �  �  �  �  k  M  4      �  �  �  �  �  �  �  �  �  �  p  Z  C  -  a  L  %    �  �  �  |  Y  6    �  �  �  s  O  ,  �  �  �        
    �  �  �  �  �  �  �  �  �  n  [  G  3      q  V  ;  (        '      �  �  �  �  {  ]  @    �  �  6  .  &       #  '  &  !      �  �  �  �  �  v  O  >  /  �  ~  s  d  P  ;  ,  #      �  �  �  �  �  �  �  y  [  =    	  �  �  �  �  �  �  �  �  t  c  R  A  .    �  �  4  �  �  �  �  �  �  �  �  �  h  @    �  �  m  4  �  �  I  �  3  T  >  (      �  �  �  �  �  �  �  �  �  �  }  d  H  ,  