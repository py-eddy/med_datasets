CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���E��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�
�   max       P�&�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =\      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @FG�z�H     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vo
=p��     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @��           �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >?|�      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0�8      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B0L<      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       > �   max       C�^A      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =���   max       C�fc      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�
�   max       PJV      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?�Ϫ͞��      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       =�x�      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @FG�z�H     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vo
=p��     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @M�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @�Q�          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?��_o�      `  U�               	                     f      O   
      R   
      
   	               �         %   5               
               *   ?                     
      
                        	   Z         3Oq�
N�bNUICN�6N���O�N�*[Nf��NB��N���M�
�P�&�O��QP��ENtQOH�,P!R�N��NMN���N�b8NV�6O�.mOv\N���P}eN�	N)M�OO};OL@ O�O�N��FNjӼNu
N=I|OR_\O���P�O��PX�PNN��(O,�OٵN�(O��JM���N�o�N��N�%�N��Nt�N�R�O�N�A�O<l�OƛN�e�O� 9NA
<N;
EO�ء���t���h�ě���9X%   %   ;o;o;D��;D��;�o;�o;��
;��
<#�
<D��<D��<D��<e`B<u<�t�<���<��
<�1<ě�<�h<�h<�h<�<�<��=+=C�=t�=�P=��=�w=�w='�=,1=,1=<j=<j=@�=P�`=P�`=Y�=Y�=aG�=e`B=ix�=m�h=u=�C�=�t�=�t�=��
=�{=�E�=�^5=\Z\ZY[gt�������~tg_\Z$'(05<@><0$$$$$$$$$$;<BMNQ[^[VNB;;;;;;;;����������������������������������������{������������������{������������������������������������������������������������ghnt|��������wtrshggefhjtu���wtsoheeeeee����)5Nft�t[7)��������#+020#
�������5N[syqhNB5����W[_gstzxtg][WWWWWWWW*-CO\bhg\OC6*'�������������		

#%(*)##"
		]`abnz��znna]]]]]]]]MNRZ[gtx���vtg[NMMMM#)/<EHUWH><6/$!�{������������������������%/:<A</#
���HGHUXanz�����znaUOHH�		"/9/*"	&**5G@[��������gB/)&%)0*)KLN[gkgf[NKKKKKKKKKK"!#+<HUahnz}naUH</$"���)669::86/)�W\hy�������������h[W�������������������()*5BGNPPNB:5)((((((UUNJH<////3<HUUUUUUU������������������������������������)6<=3)������15M[gt��tB5)��������������������������)6B>6)�����������/:=;34/)�����������������������98:;HTXacimkaWTHC<;9����	
#/1/-'#"
�����������������������y{~����������������136BCFFCBA;611111111���������	�������	
!#%+#
..,-/7:<HEHUWUUQH</.IMMH</,+/03<HMIIIIII$)66BBB=6)��������������������	)676635);;>HHMTagca\UTH;;;;;/54/.,+#
	
#/kmmimz|�������ztrok"#,/111/#!�������
������������������������������������������������������������������ÇÓàìðóòóíàÓÇ�n�e�a�X�a�n�zÇ�������������������������������������������$�)�)�)�%����������������ùŹ͹ǹù¹��������������������������A�N�Z�g�i�g�g�Z�N�A�5�.�(� �(�*�5�7�A�A²¾¿����������������¿²©¦¦ª²��������������޹޹����������
�����
� �������������������������������������������������������������������(�0�4�<�4�.�(������������s�����t�s�g�f�e�Z�Q�Z�f�o�s�s�s�s�s�sƎƚƳ���������ƶƚƆ�y�\�E�F�Q�hƎ���(�-� ��(�4�(������ؽ˽ʽ׽����ѿݿ���7�F�F�C�(���ѿ��������������������������������������������������������`�y�������������o�m�`�T�Q�L�M�P�R�T�^�`ù������������������ùìÛ��{�{�~ÈÓù�����(�2�(�"���������������<�H�L�H�B�@�<�/�,�/�/�7�<�<�<�<�<�<�<�<�m�n�y�����������z�y�x�m�k�i�k�l�m�m�m�m����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��/�;�T�d�p�o�a�T�@�"��	������"�'�$�/�zÇÈÅÇËÏÌÌÇ�~�z�n�a�^�]�h�n�z�z�;�H�T�a�a�g�k�h�a�[�T�O�H�A�;�9�;�;�;�;�[�hčďĦĳİġčā�h�B�6���%�+�6�A�[�Z�f�q�o�f�c�Z�M�L�I�M�S�Z�Z�Z�Z�Z�Z�Z�Z�нݽ߽��ݽҽнȽ˽ннннннннн���������
������������������������Y�f�r�~������������r�f�Y�U�M�I�H�K�M�Y�"�.�G�T�W�X�W�T�G�;�.�"��������"���������������޹ܹڹܹ�����������������������������������������������m�m�`�T�Q�H�R�T�`�b�m�u�m�m�m�m�m�m�m�m�H�S�R�H�B�<�/�/�,�/�<�=�H�H�H�H�H�H�H�H�F�S�_�l�x�������������l�S�F�:�-�$�+�2�F���������ʾ���� �������׾ʾþ���������"�;�T�X�V�;�/�����������������������A�M�f�s�s�s�s�n�s�}�s�f�a�M�G�A�9�7�<�A���	��"�3�7�1�)�	�����������������������#�0�I�^�n�f�U�I�0��
���������������
�#�������������������������������������������(�-�2�,�(����������������s�����������������������������s�p�l�s�s�(�-�5�A�N�Z�g�o�g�[�N�A�5�0�(����&�(�A�M�Z�f�}��������x�s�f�c�T�M�J�A�?�9�A����
�������ں���������������������	��"�$�&�"���	�����ݾ�����������ʾӾξʾ������������������������������)�5�B�N�T�[�g�s�g�[�Y�S�N�B�>�5�-�%�%�)¥¥¦²·¸²¥¥¥¥¥¥��'�*�3�:�3�3�'�'�&���������������������������������������������������e�r�~�����������������������~�l�g�e�b�e�U�Y�b�m�n�{ŀŁ�{�n�i�b�U�T�R�U�U�U�U�U�C�6�*���������������+�6�C�O�S�M�C��������������������ŭŠŞŖŔŏŔŭŹ��ǔǡǭǮǭǭǡǙǔǈ�{�x�x�{ǈǈǔǔǔǔD�D�D�D�D�D�D�D�D�D�D�D{DoDgD^DeDoD{D�D����������������������������������������Ż���!�"�(�!����������������M�Y�r���������������r�f�Y�M�@�7�9�@�M 6 4 1 N ` $  m H @ w : ^ C # U  K q [ V ; f N ` ' K < d ' E G T Q L X f v E B E ? 8 H d B o X b B F 7 ] K a T ^ O M k S 2  �  =  n  9      �  �  h  �  C  �  �    y  �  �  �  `  �  3  r  s  r      �  S  �  �  j  �  �  �  _    �  W  k  �  �  �  o  V    ;  ?  �  �    �  �  �  J  �  �  �  �  d  s  b  ��C��o�������
�T��<T��<�C�;�`B;�`B;��
;��
=��<���=�1<e`B<u=�v�<�9X<���<�j<ě�<�h=@�=o=o>?|�=t�=o=�+=��=L��=�w=#�
=<j=<j=�o=e`B=��=q��=���=���=D��=�O�=y�#=m�h=�\)=ix�=}�=m�h=��=��=�hs=�\)=��
=���=���=� �=�E�>2-=�v�=ȴ9>z�B	��B%ȥB�B23B�7B��B��B�+B"�B�kB$jB�B#��B�B	M�B0�8B0�Bv�B��B	?�B_�BLB�B��A���B	p~BoB�RB�B�sB�"Bs;B��B��B�&B��BB�B�2B*B��B�KB��A�;�B� B�jB�B�B#cB$��BڶB�BnNB ��B�A� �B��A���B�9B��BƧB)��B$cB	��B%�B��B'�Bn�B��B��B��B"��B��BK<B?�B$�(B�~B	A5B0L<B@�B��B��B	k�B�B@ B��B�A�z�B	�BȓB�B�B��B�B~(B��B��B�iB��B<PB BɯB�[B9kB3/A�x�BŐB��B�=B��B#@OB$�BÞB�BC/B ��B�xA��3B@�B @6B�]B>B�GB)�B?7Aɚ�@���A�> �A�xA��x?B��A�Ky@�v�A5b�AA�,B�5A0b'A�e9A�SGAkz�A��]A�$�AÊ�Am(AJ�zC�^AA�9�A��A�7�Aګ�A?y�A*�A҅@�J�Aa��?)��A�f�Ai7{A�Ъ@��APYTA��A?G.A�2A�ӃA���A�MfA�hOA�uuA@z�@O��AZ5AMt�A��9A��T?��+@��@	L*A�fA��^A�@�B:�C���A��@a�@߫{A��@��A���=���A��'A���?Py�A���@�`�A6��A@�>B��A0��A���A�{XAjϭA�:A��Aï�Al��AI�tC�fcA�{
A�y�A��A�)TA@ޝA*�bA��@���Aa�?0bA�x%Ah��AÀ9@��AS-�A�lDA>�A��A�A�r&A�oyA�u�A�ceAA4@THA['�AN`A��A�t�?���@���@
�A�|HA�9yA�zRB1:C���A��y@cNP@��^               
                     g      P   
      R         
   
               �         &   5      	                        +   ?                     
                     	         
   Z         3                                    =   %   ;         %                  %         3                                 !   /      +   )                                                                                                   #   %                              !         !                                 !   %      #   !                                                               O/��N�bNUICN�6N��VN}z�N�U�Nf��NB��N���M�
�O���O��4O���NtQOH�,O�#�N��NMNE06N�b8NV�6O���N��N���O��5N�	N)M�N��zO�O��}Nq��NjӼNu
N=I|O(RO���Oӯ%O��PJVO�ǥN��(O,�N�<N�J=O��5M���N�o�NDz�N�%�N��Nt�N�R�O�N�A�O<l�N��~N�e�O&gNA
<N;
EOr��  �  �  d  �  d  i  �  o  �    �  e  '  |  �  J  	b    �  d  �  b  H  �  
  �  �  �    
�  `  "    !  ?  �  �  s  �  X  .  �  M  �  �  �  4  �  >  D  �  n  5  �  j  �  �  ;  �  �  �  	��C��t���h�ě���1;��
;��
;o;o;D��;D��=aG�;��
=8Q�;��
<#�
=t�<D��<D��<�o<u<�t�<�9X<ě�<�1=��<�h<�h=0 �=#�
<��=o=+=C�=t�=#�
=��=0 �=�w=49X=m�h=,1=<j=@�=D��=T��=P�`=Y�=]/=aG�=e`B=ix�=m�h=u=�C�=�t�=��P=��
=�x�=�E�=�^5=���b_]``gr��������ytmgb$'(05<@><0$$$$$$$$$$;<BMNQ[^[VNB;;;;;;;;������������������������������������������������������������������������������������������������������������������������ghnt|��������wtrshggefhjtu���wtsoheeeeee)5KQZ[YTI5)�����#+020#�������)5BHTTQKB5�W[_gstzxtg][WWWWWWWW*-CO\bhg\OC6*'��������������		

#%(*)##"
		]`abnz��znna]]]]]]]]U[agtu~tmg_[UUUUUUUU#)/<EHUWH><6/$!�{��������������������������#(/476/
���SOU`anxz�zqnbaUSSSS�		"/9/*"	>;<AIN[g��������tNF>%)0*)KLN[gkgf[NKKKKKKKKKK.+./<HNUWUKH</......  )367764*) Y^hj{�����������th[Y��������������������()*5BGNPPNB:5)((((((UUNJH<////3<HUUUUUUU��������������������������������������)6<=3)������*0=L[_\NB5)��������������������������)6>A=6)����������).4672)����������������������98:;HTXacimkaWTHC<;9����
#*%#!
������������������������{|����������������{136BCFFCBA;611111111���������	�������

#%)#






..,-/7:<HEHUWUUQH</.IMMH</,+/03<HMIIIIII$)66BBB=6)��������������������	)676635);;>HHMTagca\UTH;;;;;/54/.,+#
	
#/lmqz���������{yutpml"#,/111/#!��������	

������������������������������������������������������������������n�zÇÓàëìíìâàÓÇ�z�m�a�\�a�g�n�������������������������������������������$�)�)�)�%����������������ùŹ͹ǹù¹��������������������������A�N�Z�g�g�g�e�Z�N�A�5�.�5�9�A�A�A�A�A�A¦²¿����������¿¾²¦¡¢¦¦¦¦¦¦���������	������������������
�����
� �������������������������������������������������������������������(�0�4�<�4�.�(������������s�����t�s�g�f�e�Z�Q�Z�f�o�s�s�s�s�s�sƳ����������������ƳƧƚƎƁ�u�k�m�uƎƳ��(�+���(�3�(������ڽ̽˽ݽ�����ݿ������&�%�!������ݿѿǿ̿ѿٿ������������������������������������������`�y�������������o�m�`�T�Q�L�M�P�R�T�^�`ìù������������������ùìàÓÍÍÑàì�����(�2�(�"���������������<�H�L�H�B�@�<�/�,�/�/�7�<�<�<�<�<�<�<�<�y�����������y�o�m�k�m�o�y�y�y�y�y�y�y�y����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��/�;�H�T�b�f�n�m�`�U�H�<�/�"���%�*�)�/�n�z�ÇÇÊÇÄ�z�n�h�a�`�a�a�l�n�n�n�n�;�H�T�a�a�g�k�h�a�[�T�O�H�A�;�9�;�;�;�;�O�[�h�tāćČĎĉā�t�l�[�B�3�3�4�>�B�O�Z�f�q�o�f�c�Z�M�L�I�M�S�Z�Z�Z�Z�Z�Z�Z�Z�нݽ߽��ݽҽнȽ˽ннннннннн���������	���������������������������f�r�v��������w�r�f�_�Y�M�L�K�M�O�Y�f�"�.�D�G�T�V�V�T�;�.�"���������"����������������ݹ���������������������������������������������������m�m�`�T�Q�H�R�T�`�b�m�u�m�m�m�m�m�m�m�m�H�S�R�H�B�<�/�/�,�/�<�=�H�H�H�H�H�H�H�H�S�_�l�p�x���������x�l�_�S�F�:�.�.�6�F�S���������ʾ���� �������׾ʾþ���������"�;�L�Q�M�;�/�"�	�����������������	��A�M�f�s�s�s�s�n�s�}�s�f�a�M�G�A�9�7�<�A���	�"�)�2�5�/�&��	����������������������#�0�<�E�W�U�D�<�0�#����������������������������������������������������������(�-�2�,�(������������������������������������������s�q�m�s�u�����(�5�A�N�Z�f�Z�Y�N�A�5�2�(��(�(�(�(�(�(�M�Z�f�|��������u�s�f�e�W�M�L�@�;�A�C�M����
�������ں���������������������	��"�$�&�"���	�����ݾ�����������¾ʾ̾ʾ������������������������������)�5�B�N�T�[�g�s�g�[�Y�S�N�B�>�5�-�%�%�)¥¥¦²·¸²¥¥¥¥¥¥��'�*�3�:�3�3�'�'�&���������������������������������������������������e�r�~�����������������������~�l�g�e�b�e�U�Y�b�m�n�{ŀŁ�{�n�i�b�U�T�R�U�U�U�U�U�C�6�*���������������+�6�C�O�S�M�C��������������ŹűŭŦŠŘŔŔŠŭŹ����ǔǡǭǮǭǭǡǙǔǈ�{�x�x�{ǈǈǔǔǔǔD�D�D�D�D�D�D�D�D�D�D�D�D{DrDoDiDoD{D�D����������������������������������������Ż���!�"�(�!����������������r�������������r�f�Y�M�D�@�9�;�@�M�Y�r 7 4 1 N J < + m H @ w 4 ` ! # U  K q K V ; [ 9 ` + K < * " B D T Q L K f b E @ 5 ? 8 / e ? o X Y B F 7 ] K a T M O 6 k S 8  �  =  n  9  �  �  �  �  h  �  C  
  �  �  y  �  �  �  `  j  3  r    �    �  �  S  �  C  *  �  �  �  _  �  �  E  k  _  �  �  o    �  *  ?  �  d    �  �  �  J  �  �    �  b  s  b  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  b  w  �  �  �  �  p  V  :    �  �  �  {  ;  �  �  1  �  �  �  �    }  z  u  f  W  H  9  '     �   �   �   �   �   �   x   a  d  `  \  X  T  Q  M  L  L  L  M  M  M  J  ?  5  *         �  �  {  v  q  j  `  V  L  B  7  ,       
  �  �  �  �  �  )  E  a  \  S  K  C  <  6  /  (  "          �  �  �  �  �  �  #  C  O  Z  c  h  c  [  Q  :  �  W  �  �  e     �   O  ,  T  e  q  y  ~  �  z  j  U  ;    �  �  �  o    �     �  o  m  j  h  e  c  a  ^  Y  Q  I  A  6  *      �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  p  g  ^  U  L  0    �  �    y  r  k  d  ]  V  O  H  A  ;  6  1  ,  '  "          �  �  �  �  �  �  �  �  �  �  �  {  o  c  X  L  @  4  (    �  �  �  �  5  T  N  ?  A  Z  [     �  �  4  �  �  �  �      $    �  �  �  �  �  r  H    �  �  p  P  9    �  i   �     �  �    J  l  w  t  s  y  {  l  N    �  ]  �  �  �  k  �  �  �  �  �  �  |  l  ]  O  B  7  0  '      	  �  �  �  J  H  F  C  A  <  7  3  -  %           �   �   �   �   �   �    �  	  	5  	O  	_  	`  	Y  	C  	  �  �  ;  �  u  �  P  )  �  3      �  �  �  �  �  �  �  l  Q  5    �  �  �  �  �  �  <  �  �  �  �  �  �  �  �  �  �  �  {  q  f  ]  T  L  J  L  M  +  B  Y  _  c  ]  U  C  -    �  �  �  �  f  A    �  �  y  �  �  �  �  �    s  b  N  7    �  �  �  �  �  �  ~  |  z  b  R  @  *    �  �  �  �  �    a  A  %    �  �  �  $  �  7  2  G  ?  9  %    �  �  l  5    �  �  z  6  �  ^  9    �  �  �  �  �  �  �  �  �  �  r  0  �  �  e  .  �  �  �  �  
  	      �  �  �  �  �  �  �  �  �  t  T  -    �  �  �    �  !  �    ]  �  �  �  �  z  "  �    \  �    	�  �  2  �  �  �  �  �  |  s  j  a  W  M  C  5  (  '  8  H  K  I  F  �  �  �  �  �  �  �  �  �  �  �  �  �  �        )  6  C  �  �  S  �  �  �        �  �  �  Y    �  C  �  0  �  "  
Z  
  
�  
�  
�  
�  
~  
C  	�  	�  	_  �  �  3  �  G  ]  .  �  `  [  _  Y  Q  M  F  =  2  (      �  �  �  �  Z    �  W   �        !      
  �  �  �  �  �  �  �  �  _  5    �  �      �  �  �  �  �  {  ^  6    �  �  �  Y  /    �  �  �  !  7  A  8  ,       �  �  �  �  �  i  E    �  �  �  �  �  ?  7  .  $      �  �  �  �  �  �  Q    �  �  W    �  �    �  �  �  �  �  �  r  X  :       �  �  �  �  �  h  G  �  �  �  �  �  �  �  u  d  Q  :       �  �  �  m  B          W  k  r  m  a  Q  >  )    �  �  �  �  m  #  �  �  Y  �  �  �  �  �  �  y  ]  =    �  �  �  �  q  <    �  �  t  �  R  R  W  T  M  <    
      �  �  �  @  �  �    �  �   �  �        *  .  &       �  �  ~  ?  �  �  !  �  �  �  .  �  �  �  �  �  �  �  �  }  p  a  O  >  )    �  �  �  �  �  M  8  "  	  �  �  �  x  L    �  �  s  /  �  �  (  �  I  �  h  �  �  {  f  M  3    �  �  �  �  Z  +  �  �  O  �  b   �  �  �  �  �  n  O  +    �  �  n  !  �  �  :     �  �  A   �  �  �  �  �  �  s  l  e  U  ?  (    �  �  �  �  �  �    %  4  5  6  7  )      �  �  �  �  x  I    �  �  �  J    �  �  �  �  �  �  �  |  q  g  [  O  >  &    �  �  �  �  �  �  %  +  2  8  9  $    �  �  �  }  O    �  �  �  N    �  �  D  (    �  �  �  �  �  �  �  �  r  N  )  �  �  w  F     �  �  i  C  �  �  i  "  �  �  �  �  �  �  �  ~  _  :    �  �  n  a  K  .    �  �  �  �  f  C     �  �  �  F    �  [  �  5  4  1  -  '           �  �  �  �  �  �  y  Y  I  J  X  �  �  �  �  i  M  -  
  �  �  �  v  N    �  9  �  (  �  �  j  ^  Q  @  ,    �  �  �  �  �  �  l  E    �  �  �  x  L  �  �  s  L  -      �  �  �  �  ~  c  E  !  �  �  ~  }  �  �  �  �  �  �  �  �  �  �  s  \  C  &  	  �  �  �  �  �  `  ;    �  �  �  �  �  a  =    �  �  �  d  .  �  �  �  3   �  �  C    `  �  �  �  �  �  k    �  �    1  .  
  	!    u  �  �  �  �  �  �  �  �  �  �  l  F     �  �  �  �  t  U  6  �  �  y  e  L  4      �  �  �  �  g  J  0      �  �  �  	�  	�  	�  	|  	T  	!  �  �  9  �  f  �  [  �  W    �  z    �