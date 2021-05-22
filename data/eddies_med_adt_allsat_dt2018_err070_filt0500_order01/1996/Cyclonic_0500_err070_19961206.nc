CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��l�C��        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N	��   max       P���        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       <��
        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @E�\(�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vLQ��     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�Y�            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <u        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�x)   max       B+��        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B+��        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >E��   max       C�?)        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       > ��   max       C�H	        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          m        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�N   max       P���        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u%F   max       ?�<64�        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       <��
        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @E�\(�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vLQ��     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @N            �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @��             U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�a��e��        W�   3                                              !         m               L   9   6      	            
            -                     
                           #      	         2   
   
               O�ݳNRF�Nx�UNR,&N�4�O�(�O�8�N��N+{�N	��N3 N��OhTN� ZN吏O��EOpX'N���P���O��>OZXO�xEN��P���Pn�GO�z�O���N@x�N�N0OM�OC�O�&�O1n�O�P\�nO!8�O��O�NOc^�NJ��Nj��O	��O%�N+�lO]<�NCzN�T�O��,O�9�N��qO�m6N�f�O/�LO��N��gO�_#OFN5O��O�cN��O��OdA N�2�<��
<�t�<�C�<e`B<49X<#�
<t�<t�;D��:�o��o�ě��o�#�
�49X�49X�e`B�e`B�u��t����㼬1��1��9X��j������/��/��/��h���o�C��t�������w�#�
�,1�,1�49X�49X�8Q�@��@��H�9�H�9�H�9�T���T���Y��]/�aG��aG��y�#�����7L��7L��t�������{��E���E��oUH</#	!/<HPVWVWU��������������������}������������z}}}}}}tzz�������}ztttttttt������������������������������������������������������jnqz���������zyrokkjagtxtige^aaaaaaaaaa?BOQ[b[OBB??????????��" �������+6ORX]_[YOB62)O[bhltz����th_[XVSOO��������������������	
#/HUX^URL</#	��������������������y{��������~{vuyyyyyy���������������������������������������������������������
#%/2/(
��������������������������<Qaz�����������oUH;<y������������yyciu�������������thicHamz������zrmaTHC>BH./<BHQHF<3/*........7<EHNSOUWUUKH<:<<<77Y[hjnjh[VSYYYYYYYYYYgmz�����������zmhccgbqt������������tphcb)5ANPJB5)	
���������������������)575*)	 ���[int�����������aVNN[NT]almpwzzvmaTSMJJN()05BNX\g[VPONLB5)'(z���������������|yxz�����������������������������������������
#(%#
	������LOT[]]hontqnhe[OKEEL����������������������������������������������������������������������������������������������������#0<GMOIIMIF<0!\bn���������{njec_\\Ybchn{���{rnb\YYYYYYU[ct�������th[UQORRU|����������������|||��������������������cgnt����������tnkkccNUadntz���ztnfaWUPNN���" !���������������+01<HINSUWVWUIA<70,+`ainz�������zsnea_^`��������������������s|������������tjikps�������� �����������
#+##
�����������������������ŹŭŤţŧŭŹ����������
������!�&�)�&�!���������ּӼּټ����� ��������ּּּּּ�������ýû�������������������������������g�f�Z�Y�Q�O�Z�g�s�v�t�s�o�h�g�g�g�g�g�gà×ÏËÃ�yÀÅÇÓàçð��������ùìà�H�/��� � �	��#�/�<�B�H�Q�Y�c�h�a�U�H�A�9�A�E�I�N�Z�a�g�n�s���������s�g�Z�N�A�����������������������������������������a�^�`�a�h�n�y�x�n�n�a�a�a�a�a�a�a�a�a�a�Z�Z�Q�Z�a�g�i�s�w�u�s�g�Z�Z�Z�Z�Z�Z�Z�Z���	�
�����)�/�5�6�5�)�(������l�b�S�O�P�S�_�l�x�������������������x�l�L�B�L�P�Y�a�e�l�r�~�������~�{�r�e�Y�M�LŔŒŔśŠťŭŹ����������ŹŭŠŔŔŔŔ���������)�<�B�O�f�i�Y�X�O�B�6��r�e�Y�Q�L�@�<�@�P�Y�e�r�~�������������r�F�C�C�F�N�S�_�l�s�v�l�h�_�S�F�F�F�F�F�F�����a�j�~�����л��'�M�`�Z�B�&����л����x�x�s�l�x�������ûлܻ������лû����U�N�H�=�<�3�3�<�H�U�a�i�n�n�o�n�l�a�_�UìàÚÑÌÓÛìù������������������ùì�M�C�D�A�=�A�I�M�Z�f�k�m�f�Z�M�M�M�M�M�M�����g�a�o�n�_�k�x���ܺ�@�E�:����Ϲ������y�\�I�G�L�N�T�`�m�����Ŀ���������Y�M�@�4����'�@�M�Y�f�r�����������r�Y�(������������5�K�Z�j�������s�N�5�(�H�G�@�H�R�U�Z�a�e�a�`�U�H�H�H�H�H�H�H�H���
��
��#�)�/�<�=�<�;�5�/�%�#�"�����������ùϹҹչϹù���������������������������������������������
���������������¹º¿����������������� ����������Ň�{�s�r�{ŇşŠŭŵź����������ŭŠŔŇ�����(�4�4�A�F�E�A�>�4�(�������"������	���"�$�2�;�E�G�R�G�>�.�"�a�/��	�����	�"�;�T�a�o�������������z�a������������������	�������	����������������������������������������������ƍƁ�u�h�e�e�l�uƁƚƳ����������ƳƧƚƍ���������������������$�'�)�'�%�������{�z�o�b�_�b�f�o�x�{ǈǏǈ�~�{�{�{�{�{�{�!��!�'�*�-�:�<�A�F�J�F�:�-�!�!�!�!�!�!�m�k�m�n�m�`�[�`�m�y�������������������m�"��	���������	��"�/�;�H�L�T�I�H�;�/�"����������������������������������������������������������������������������������ֺܺӺϺֺ���������������ɺɺ��������������������������Ǻɺʺɺɻ��������ûлܻ�����������ܻлû����	���'�4�@�M�Y�_�j�m�n�f�Y�M�4�'��s�h�g�Z�T�Q�Z�g�s�y���������s�s�s�s�s�s�������������	��"�.�6�;�=�:�/�"�������ŹůŭšŠśŠŭŹ������������������ŹŹ�;�7�;�>�B�?�G�T�W�`�d�m�n�r�p�m�`�T�G�;���������������������+�2�A�5�)������������������ĿĿĿʿѿտѿͿĿ��������!����������!�:�S�l���������l�`�G�.�!��ݽнĽ������Ľн��������
����齫�����������������Ľнݽ���ݽؽн������׾Ҿʾƾ¾ʾ;׾����	��	����������O�F�M�O�T�[�h�t�{�{�t�k�h�[�O�O�O�O�O�O�����������������������
��%�%�#��
�������������������������������
�������D�D�D�D�D�D�D�D�D�D�D�E EEEED�D�D�D�  . f D a _ Q G N k 4 > = e 2 < * 3 F h + 3 K < T ( x 4 R @ [ > 1  r 8 E N 6 \ L Z ; ` @ ! 8 e > ! J J O F D f O A ^ & O Q ! i  \  g  �  r  �  �  �  $  @  Q  H  �  �  6    M  �  �  '  �  1  �  �  �  e    �  ]  �  1  �  �    p  {  �  q  Y  y  ,  Y  �  +  �  M  �  ]  �  c  A  �    �  �  V    �  �  D  Z  �  I  �  ���<u;��
<49X;�`B��1�T����o:�o���
�ě��T������h��t��0 Ž\)��1���m�8Q�49X�<j��/�ȴ9���
���
�aG��\)�C��#�
�L�ͽ,1�q���ixսH�9����P�`�P�`����}�m�h�@��aG��}�Y��y�#�]/�q��������ixս�E��}󶽁%���w�����h���㽧������`��h���Bt�B+�FB�B�yB��B1B4�ByB	�B��B��BhB�qB��B�dBY�B!�HB)=�B:�B��B!M�B��B"}�B�B+��B�gA��	B�!BK�Bv<A���B��B��BгB�`B'�A�x)B�|BBѬB
��B$�BNBSB�B�BW�BǠB%��B)�B(:ZB{B
�$B�kB
��B��B�[B�)B&u�B&B�fB
��B#7B�B?�B+��B:FB��B�B��BHUB��B	LSB4�B�]BA�B�;B��B��B>�B"@=B)D�B��B 7�B!4sB�ZB"S�B?9B*�wBD�A��BQ�B=PB@fA��B_B��B?OBQ�B�OA�z�BA�B<mBI�B
��B$L1B@�B?�B�MB�1BD�B�B%CB(�:B(n�B?�B
��B2�B
��B@&BHYB��B&@�BʾB�fB
��B@B�2A�l�@f��A��A�=A�:�AˉA��8A�6�A�#�A�2kA�h�A�Bx@�)�?� A���A��?�U@��y@�Q@���A�_kA��A>��>���As�@ٳ�A��pA�]�A���>E��A�<�A�<�A��AA6�A_��A�|qA�(�A��=B6�B��B�@t�QAn��A���A�JLA���@Br�@8@�)�@ОA��A��-A���Af�A���Av�Am2A,aYA&�AU�sA�,�A漛A�O�C�?)A��@j��AnA��A�~A�S%A�A�yPA�Y�A��A��pA�`C@���?�"�A��A׀�?�h�@�?�@���@�>A�4�A�%A>��> ��Ar�t@��wA��7A�~gA��A>@<A�<A���A�oA6��Aa_A���A���A��FBƁB	 �BC5@s��Am
�A�x�A�u A�R@D��@��@�MW@��XA��A�zWA���Ag�A�Ax��A:�A.��A&�7AU4rA�psA��A懬C�H	   3                                              !         m               M   :   7      	                        .                                                $      	      	   2   
                                                                           C         !      =   9   '   +                           1                                       !         #               )                                                                              %               =   9   %   +                           1                                                #               )                     O�ݳNRF�Nx�UNR,&N�4�OL,�O��N���N+{�N	��N3 N��O;d3N��N吏OA�O_e�Nl��P��O��>OZXO���N9uBP���Pn�GO�eO���N@x�N�M�NN�}OC�O0|wO1n�Nߩ�P\�nO!8�O��O�GOP��NJ��Nj��O	��O%�N+�lO]<�NCzN�
O���O
�IN��qO�m6N�f�O/�LO��N��gO�_#OFN5O��O�cNC��Onq�OJ�KN�2�  	  �    i    �  �  �  �  �    �  �  �  �  �  �  �  �  ]  C  �  �  o       �    �  f  v    4  �  T  N  b  �  W  +  9    �  B  l  e  �  �  �  �  �  '  e  �    �  b  �  z  A  �  �     
�<��
<�t�<�C�<e`B<49X;�o:�o;�`B;D��:�o��o�ě��49X��C��49X��9X�u�u�y�#��t����������9X��9X��j��`B��/��/��/���t��o��w�t���w����w�#�
�49X�0 Ž49X�49X�8Q�@��@��H�9�H�9�L�ͽY���%�Y��]/�aG��aG��y�#�����7L��7L��t�������Q콺^5��^5�oUH</#	!/<HPVWVWU��������������������}������������z}}}}}}tzz�������}ztttttttt���������������������������������������������������������nnnz������ztqnnnnnnnagtxtige^aaaaaaaaaa?BOQ[b[OBB??????????��" �������")06OOVZ[[OB6)[[hhtu����thg[[Z[[[[��������������������#/<HMNKHD=<3/#!��������������������w{����������{wwwwwww����������������������������������������������������������������
#*.,$
�������������������������<Qaz�����������oUH;<y������������yyekv�������������|jkeHamz������zrmaTHC>BH./<BHQHF<3/*........7<EHNSOUWUUKH<:<<<77Z[hhlhh[XUZZZZZZZZZZlmz��������zmlggllllbqt������������tphcb).5=BEJCB5)�����������������������)2))����[int�����������aVNN[NT]almpwzzvmaTSMJJN()05BNX\g[VPONLB5)'(|����������������|z|�����������������������������������������
#(%#
	������LOT[]]hontqnhe[OKEEL����������������������������������������������������������������������������������������������������#0<FLMIHIE<0"ln{����������{unmijlYbchn{���{rnb\YYYYYYU[ct�������th[UQORRU|����������������|||��������������������cgnt����������tnkkccNUadntz���ztnfaWUPNN���" !���������������+01<HINSUWVWUIA<70,+`ainz�������zsnea_^`��������������������rt}�������������tkmr��������� �����������
#+##
�����������������������ŹŭŤţŧŭŹ����������
������!�&�)�&�!���������ּӼּټ����� ��������ּּּּּ�������ýû�������������������������������g�f�Z�Y�Q�O�Z�g�s�v�t�s�o�h�g�g�g�g�g�gàÛÓÑÏÉÈÃÉÓàéìù������ùìà�<�2�0�/�$�'�+�/�<�@�H�L�N�Q�U�X�U�H�<�<�Z�X�N�I�L�N�Z�g�s��~�s�g�Z�Z�Z�Z�Z�Z�Z�����������������������������������������a�^�`�a�h�n�y�x�n�n�a�a�a�a�a�a�a�a�a�a�Z�Z�Q�Z�a�g�i�s�w�u�s�g�Z�Z�Z�Z�Z�Z�Z�Z���	�
�����)�/�5�6�5�)�(������l�f�_�X�S�Q�R�S�_�l�v�y�������������x�l�Y�Y�X�Y�e�e�q�r�}�~����~�t�r�e�Y�Y�Y�YŔŒŔśŠťŭŹ����������ŹŭŠŔŔŔŔ�)��!�"�&�)�6�B�G�O�[�]�[�Z�O�O�B�6�)�)�r�e�Y�R�L�@�@�L�Y�e�r�~�������������~�r�S�I�F�F�F�O�S�_�l�p�t�l�f�_�S�S�S�S�S�S�����������������ܻ����&�!�����ܻл����x�x�s�l�x�������ûлܻ������лû����U�N�H�=�<�3�3�<�H�U�a�i�n�n�o�n�l�a�_�U��ùìàØÕÙàìù�������������������žM�E�F�M�Z�f�j�k�f�Z�M�M�M�M�M�M�M�M�M�M�����g�a�o�n�_�k�x���ܺ�@�E�:����Ϲ������y�\�I�G�L�N�T�`�m�����Ŀ���������Y�M�@�4����'�4�@�M�Y�f����������r�Y�(������������5�K�Z�j�������s�N�5�(�H�G�@�H�R�U�Z�a�e�a�`�U�H�H�H�H�H�H�H�H���
��
��#�)�/�<�=�<�;�5�/�%�#�"�����������ùϹйӹϹù�������������������������������������� ��� ��������������������¹º¿����������������� ����������ŠŔŇ�{�w�w�{�ŇŎŔŠŭŮŵŶŶųŭŠ�����(�4�4�A�F�E�A�>�4�(�������.�%�"�������"�"�.�0�;�C�G�;�:�.�.�a�/��	�����	�"�;�T�a�o�������������z�a������������������	�������	����������������������������������������������ƎƁ�u�h�h�p�uƁƎƚƳƾ��������ƳƧƜƎ�������������������$�%�)�'�$��������{�z�o�b�_�b�f�o�x�{ǈǏǈ�~�{�{�{�{�{�{�!��!�'�*�-�:�<�A�F�J�F�:�-�!�!�!�!�!�!�m�k�m�n�m�`�[�`�m�y�������������������m�"��	���������	��"�/�;�H�L�T�I�H�;�/�"����������������������������������������������������������������������������������ֺܺӺϺֺ��������������⺋�����������������ĺ����������������������������ûлܻ����������ܻлû��'�&�����(�4�@�M�P�Y�_�]�Y�R�M�@�4�'�s�h�g�Z�T�Q�Z�g�s�y���������s�s�s�s�s�s�������������	��"�.�6�;�=�:�/�"�������ŹůŭšŠśŠŭŹ������������������ŹŹ�;�7�;�>�B�?�G�T�W�`�d�m�n�r�p�m�`�T�G�;���������������������+�2�A�5�)������������������ĿĿĿʿѿտѿͿĿ��������!����������!�:�S�l���������l�`�G�.�!��ݽнĽ������Ľн��������
����齫�����������������Ľнݽ���ݽؽн������׾Ҿʾƾ¾ʾ;׾����	��	����������[�X�R�Y�[�h�t�u�u�t�h�]�[�[�[�[�[�[�[�[���������������������������
��#�#��
�����������������������������
��������D�D�D�D�D�D�D�D�D�D�D�E EEEED�D�D�D�  . f D a d a 4 N k 4 > > R 2 ( " ( ; h + + ( < T & x 4 R , 6 > )  ^ 8 E N 0 \ L Z ; ` @ ! 8 \ ?  J J O F D f O A ^ & ? P  i  \  g  �  r  �  �  Y  �  @  Q  H  �  �  �    Q  �  u  r  �  1  -  O  �  e  �  �  ]  �    �  �  y  p  $  �  q  Y  !  �  Y  �  +  �  M  �  ]  �  :  +  �    �  �  V    �  �  D  Z  h    �  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  	  �  �  �  �  V  #  �  �  d    �  �  (  �  \  �  #  q  �  �  �  �  �  �  �  �  �  �  ~  x  s  m  e  X  K  >  2  %          �  �  �  �  �  �  �  �  �    c  F  !  �  �  �  d  i  `  V  M  D  :  1  '        �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  }  n  \  H  5  !    �  �  �  �  �  �  �  �    V  !    �  {  7    �  �  d  �  �  �  w  j  `  ~  �  �  �  �  �  �  �  �  �  y  f  �  '  �  ]  y  �  �  �  |  n  \  I  4      �  �  �  x  (  �  d   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  l  M  $  �  �  �  �  �  {  l  _  b  e  h                    �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  }  s  h  d  b  _  ]  Z  L  /    �  v    �  �  �  �  �  �  v  U  3    �  �  �  �  �  g  4  �  �  �  �  �  �  �  �  �  �  �  �  �  t  9  �  �  j    �  u  7  �  �  �  �  �  �  �  �  �  �  �  |  q  f  [  P  E  :    �  �  +  <  M  `  q  �  �  �  �  m  J    �  �    2  �  m  �  [  �  �  �  �  �  �  �    d  O  ;  $    �  �  �  �  l  ?  �  �  �  �  �  �  �  �    z  m  ^  M  1    �  �  �  k  )   �  b  �  �  �  �  �  �  �  �  �  �  �  �  .  �  �    &  �    ]  G  .    �  �  �  �  �      �  �  �  �    5  �  �  a  C  @  8  1        �  �  �  �  }  P    �  �  @  �  L  �  �  �  �  �  �  �  �  �  �  k  J  $  �  �  �  V  �  g  	  �  ~  �  �  �  �  �  �  �  w  j  Z  G  3       �  �  �  �  �  o  L    �  �  c  �  �  �  �  ^    �  @  �  Z  �    �  ]      �  �  �  �  �  �  o  <    �  �  P    �  3  �  �   �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  f  �      �  �  �  �  �  �  �  �  `  /  �  �  y  L  %    �  g  �  |  "    ,  ;  C  E  G  F  D  A  =  7  .  %      �  �  �  o  F  �  �  �  �  �  �  z  b  J  M  U  [  W  S  N  J  F  B  =  9  F  R  \  c  g  g  X  E  (  	  �  �  �  r  M  *    �  �  �  �    G  c  n  v  u  q  e  V  C  .    �  �  H  �  K  �      �  �  �  �  �  �    c  B    �  �  �  �  ^  D  ,      �  
  (  2  2  *      �  �  �  �  L    �  {    j  �    �  �  �  q  V  6    �  �  y  E    �  �  G  �  �  (  �  +  R  S  M  3  '  '    �  �  �  �  �  �  �  �  ]  3  	  �  �  N  H  .    �  �  O    �  �  �  n  d  ^  T  ;    �    v  b  Y  R  K  J  J  G  <  -    �  �  �  �  x  D  
  �  v    �  �  �  �  �  �  �  �  t  E      *    �  �  �  �  �  u  A  O  V  Q  J  A  4  !    �  �  �  �  Y    �  |  +  �  5  )  *  $    
  �  �  �  �  �  �  i  K  *    �  �  e  ,  	  9  =  &  �  �  �  y  D    �  �  I  �  �  Y  �  �  1  �  c          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  j  `  Z  Q  F  7  $    �  �  �  �  �  �  �  B  0  %    �  �      �  �  �  �  [    �  e  �  �     �  l  T  =  %  )  3  =  ;  4  ,  "      �  �  �  �  �  �  �  e  [  Q  F  :  +    �  �  �  �  �  �  �    f  L  .  &  -  �  �  �  �  r  b  R  B  3  (      �  �  �  �  �  r  R  2  �  �  �  �  x  e  R  >  )    �  �  �  �  �  �  |  d  I  .  �  �  �  �  �  |  r  i  ]  M  ,  �  �  �  k  H  ,    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  J  �  �  :  �   �  �  �  �  �  �  �  �  �  �  �  ~  p  a  R  C  3  $       �  '    �  �  �  �    O  $  �  �  �  >  �  �  T  �  \  �   �  e  \  S  G  7  '    �  �  �  �  �  �  n  I  #  �  �  �  k  �  z  m  a  U  L  O  R  J  @  5  (      �  �  �  �  �  j       �  �  �  �  �  �  p  b  @    �  �  �  [    �  z    �  �  }  g  R  A  8  0  &    
  �  �  �  �  �  �  �  #  s  b  _  S  D  (  �  �  _    �  S  �  �  7  �  �  7  �  N  �  �  �  �  {  f  O  6      �  �  �  �  v  Y  <       �  �  z  a  H  -    �  �  �    ^  :    �  �  |  H  %    '  O  A  0    �  �  �  �  b  0    �  �  k  8    �  �  b  2    �  �  �  �  �  �  �  �  �  �  �  �  �  u  p  s  x  y  v  u  �  �  �  �  �  |  X  .    �  �  h  *  �  �  5  �  �  W   �  �        �  �  �  �  �  �  }  U  (  �  �  d  �  y  �  �   z  
�  
U  
%  
�  
S  	�  	�  	;  �  j  �  �  5  �  �    S  �  �  �