CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�dZ�1       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��=   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �z�   max       <�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?O\(�   max       @FAG�z�     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vz�G�{     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P�           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       <��
       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��G   max       B0�       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�v   max       B0 �       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�
T   max       C��       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          k       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��=   max       P��       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�'�/�   max       ?����-�       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �z�   max       <�       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?O\(�   max       @F'�z�H     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vz�G�{     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P@           �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A{   max         A{       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?����-�     �  Zh               6   N      	      $            
                              	         /         A         
      &      $      k         *   	   H               "         %         :      	      3   ,   ,                  N�N�N�)TN_[�N��vO��IPs��OøN�i�OFO���N��O�h�N��N��O�GoM��=NiF�N^�O�Z#N̥�P>��N�aN�ĩN�k#N��rO9rP��N�3�P$�{P�F*O-z�NVaN\��O�NO��kOY�[O�cNed�P�}�O�}N��Pb.O�4PUJO+�N��{O��dO���O�(�O+�N�ZwON��N�?OrpP�)N�Nw��OB�PF�kO���P��NѶO�gN�1�M�=�NU��N�R�<�<ě�<ě�<u<e`B<e`B<49X;ě�;o;o��o�o�D����o��o�ě��ě��o�o�t��49X�49X��C���C���C���C����㼴9X��9X��j�ě����ͼ�/��`B���o�+�+�C��\)�\)�t��#�
�',1�0 Ž@��L�ͽL�ͽY��Y��Y��aG��aG��e`B�e`B�ixս�%��O߽�\)����� Ž� Ž�9X�����`B�z����������������������������������������������������������������

"
������������������������������������,8:6������U_bd_al���������nUSU"#,/9<<<8/###!""""������������lo����������������~llt�����ztkllllllllll6BOTS\tqmheOB1-,,**6����������������������������������������
#HUXPH<0#
���� ����������������������������������ABEO[[][SOMB<9AAAAAA*6CPX]`cXC:*���������������������![gt������toeB��������	�����������������������������'%������������	�������������������������������#>b{�����zT<0
���HHHTUX^[TH?;=BHHHHHHg�������������zlefcg���������#��������������������������������������

�������������%/8BO[hmhbc[OB6-)!"%�������"'*%������\amonqz����znla^_^^\�����������������������

�����������h�������������~zujehlnz��������������zol�������������������������������������zu���������������������ht�������th_OA:8>O[h����������������������������������������������������������))-+5BN[gmmklg[NB6,)s������������tijlkmsW[gt���������tga[WSW*/6<HLQROH</.*******|������������������|��������������������9<HTU\ablnsb_UH?<339Uanz���������naULKLU����������������������������������������FQ[hipstttqh[YWSOKFFLU^gt���������g\NB=L)5BNSTNB;.���Xdt�����������tg_ZTX������������������������������������uu������������<<AIKII@<;69<<<<<<<<#'+/6<=</#"HHRUaakla`UHFEHHHHHHE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦©²³¶²²¦����������������������������������������E*E"E"E*E*E7E:ECEPEUEUEQEPECE7E/E*E*E*E*�/�#�����#�<�H�U�n�r�z�z�v�n�a�U�H�/���Z�A�5�(���,�A�Z�g�������������������������g�Z�N�Z�c�l�s���������������������<�:�<�<�H�R�U�W�a�l�n�n�n�a�U�H�<�<�<�<���������������������������������U�H�<�%�#�,�<�L�U�a�j�zÇÍÍ�{�}�z�a�U�H�B�H�H�U�a�b�a�_�U�H�H�H�H�H�H�H�H�H�H�X�W�N�T�`�m�����������������������y�m�X�H�G�<�:�0�4�<�D�H�I�U�[�Y�\�[�U�H�H�H�H�`�`�c�k�l�n�y��������������y�p�l�`�`�`�����'��)�6�O�l�t�n�h�e�[�O�B�6�)�àÞàèìùþüùìàààààààààà�������������������������������������U�T�U�W�a�d�n�q�z�~�z�t�n�a�U�U�U�U�U�U�;�.�������ݾ۾����	�"�.�8�;�?�F�A�;ÓÓÓÓ×àìù��������þùìàÓÓÓÓ�S�N�`�v����i�l���������ݿѿ������m�S������޺غֺݺ����������������������(�-�4�6�4�,�(�������������x�s�p�h�g�g�g�s�����������������������Ŀ������������Ŀѿܿݿ�߿ݿѿͿĿĿĿļM�G�@�8�4�9�@�C�M�Y�f�g�p�s�w�t�r�f�Y�M���{�g�N�<�5�7�Y�s��������������������������������������������������������������������������������'�4�6�4�1�4�"�	�����˺������Y�@�-�(�3�Y���ɺ�� �.�0���ֺ��׾;ʾþʾʾ׾����	�����	�����׻����x�q�l�j�j�l�p�x���������������������������������żɼ����������������������������x�l�t�����������û˻ͻ̻ɻû��������ʼ����������ʼ�����!�0�1�-�����׼ʿĿ����Ŀѿݿ������,�5�5�(�����ݿ�����ưƳƶ���������$�0�/�1�����������=�2�0�$�+�0�=�F�I�P�J�I�=�=�=�=�=�=�=�=�������������3�������ƺź����T�@�'����U�I�4�0�'�(�#���0�I�U�b�n�~ń�y�j�b�U�ѿпĿ������������Ŀƿѿֿݿ��ݿѿѿ��a�=�,�%�#�)�9�H�a�m�����������������z�a������������������������������	�����깑�������ùϹܺ��&�#�� ����ܹù�����ŠřŔŌőŔŠŧŭŹ������������ŹŭŠŠčĂĈċčĚĞĦĨĳĵĳĦĚčččččč��߻�޻�����'�*�1�3�1�'��������*����������������*�6�C�G�M�P�K�C�6�*ŔŃ�ŊŠŹ����������������������ŹŭŔ��������������	�������	����FF E�E�E�E�FFF$F/F0F&F$FFFFFFF�Ŀ��������������Ŀѿݿ������ݿٿѿ���������������� ���$�%�(�*�$�������𿟿������������������ĿѿѿпɿĿ¿������>�5�/�/�8�H�T�a�m�z�������������z�a�T�>�l�e�d�l�x�~�����������x�l�l�l�l�l�l�l�l�������������������������������������������������������������������������������������u�t�|¦¿��������!�/�9�5�#��ïáÚÜçñêìùú������������������ïĿĳĪĦğĥĳĿ���������
���������Ŀ�!���!�.�:�B�:�4�.�!�!�!�!�!�!�!�!�!�!�y�`�S�S�b�v�y�����������ĽŽĽ��������yììùý��þùìäàÛ×àììììììì��������������������������������DbDWDbDoDrD{D�D�D�D�D�D�D{DoDbDbDbDbDbDbE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� U Q K \ ) 6 D ? [ 0 C , Q , B K N 3 = & � a " B 4 7 f P  l S \ " # f v @ * H " H 5 ] ) $ = & ) & E % % l B $ } d b c I 0 ^ B L M | ,    �  �  �  �     �  �  �  y  �  8  b  �  �  i  ,  �  u  �  �  �  �  �    �  �  ]  �  �  �  �  �  b    �  N  :  j  �  �    m  Q  n  4  �  )  m  �  �  �  �  u  M  �  G  �  �    r  �  =  G  �  +  �  �<���<T��<��
�D���#�
��%�t��o�#�
�C���o�T����C��T����`B�o�o����\)��j�\)���ͼě����ͼ�h�<j��������49X��-��w�o��P�@���C��<j��O߽�w�\)�}�,1���w�D����l��aG��L�ͽ�t���hs���罙�������E��}󶽑hs��l��q�������-��F��l�������G���/�Ƨ���m�+B�B��Bc�B��B]�B�)B��B�Bq`B6�Bf�B��B ՁB*��B�BTBkhB��B0�B�jB�nB�^BЙB�ZB�B �lB&dA��GB ��B"�BJ�B ��B$Bi�B-�]A��lB��BBB*BpB*B{BNB��Bd�BGB!�B%�B
��B	�/B�B� B�B�GB|B��B4�BhMB	�B1kB
�
B��BBB
�	B&\4B�B�B<B��B�B�B9B4�B��B3�B?aB>�B�mB��B �0B+)\B6	B GB�}B�iB0 �B�B��B;BĂB�B��B ��B&��A�vB �}B#6�B<B �?B$>�B��B.?�A���B� B?dB@�B��B+�B;NB�BA�BVB�=B?�B��B
�0B	�vB��B9*BԄB@�BψB�TBT�B?�B
<�B'�B
��B�]B��B
�<B&DB@5B��C�5�A��LA�\�C��"A�PlA���A�	lA�[�A��A�MA�E�Ao�lA��%AxAؚ�A̍A�hA���A[�#A�`�Ao��@Gm�A4�BA�q�Az7a@ٴ�A�&A��rA���@'�qAV+�@�J�@��T@�H�AۆA��B�B
��?�?A�f�Ay@�A��A�<�>�
TA���Aߜ�@�3eA�|HA�q3AY�C��A{
�Bn+AvO�A��@�E�A���A�1A��Aθ]A�eKA��Ax:A̪�A1��C��eC��C�9�A�"A�t�C���A�}�A��PA�%JA�yuA��_AŃ[AŁ)Ap׬AĂ�Ah0A�b�A�y�A�}"A��AZ:�À�An�i@D<BA5@�A��:Ay��@��OA�A�A��A��>@4I^ATA�@���@��@��>A �A���B��B
��@�AAy�A��'A�0/>��A�FwA߇�@� RB =�A��-AZ��C���A{[�B��At��A���@��cA��TA��+A��A·A�r�AgA ��A�~�A2�pC���C�               7   N      	      %            
                              	         /         A               '      %      k         +   	   I               "         %         :      	      3   ,   -   	                              !   1   %         !                           !      =                  A      %   E               /      %      9   #      %      '                                 %            3      )                                                                                 %                  A      #   ;               %      !      +         %                                       !            #      )                  N�N�N�)TN_[�N��vO�XO���O��N_�+N��N�͕N��Os�MN09�N��OU��M��=NiF�NCbO3�)N̥�Oё�N_|+N�ĩN�k#N��rO9rP��N�3�P��Po`N�kCNVaN\��O�NOɕO�UO�iNed�P-IO���N��Pb.O�4O��O+�N��{O�f�O���Os0fO�N�ZwO%��N�?OrpO�oN�Nw��OB�Oݯ'Oki�P��NѶO�d�N�1�M�=�NU��N�R�  �  Q  *  )  k    ?  ^  p  5  =  0    �  �  �  �  �  �  "  �  �  s  �  �  ~    �       R  �  �  [  U  �  K  �  
�  �    ^  �  �  �  �  o  U  �  M  C  �  y  .  �  �  �  L  �  	!  �  t  �  ,  j  �  	_<�<ě�<ě�<u;�o��9X<#�
;��
��o���
��o��o�ě���o�o�ě��ě��t���t��t���C��e`B��C���C���C���C����㼴9X���ͼ�`B��h���ͼ�/��`B�C��C��\)�+��%�t��\)�t��#�
��hs�,1�0 ŽD���L�ͽm�h�]/�Y��m�h�aG��aG�����e`B�ixս�%���罗�P����� Ž�-��9X�����`B�z����������������������������������������������������������������

"
���������������������������������

��������`cdgnz���������naVW`#/7;6/#"������������������������������������lt�����ztkllllllllll-6BOchnnlhe[OB61..--����������������������������������������#/DHLHEA</%#���� ����������������������������������:BOZ[[[PONB=::::::::$*6CKOOJCB6*��������������������5Bat}xtg[NB5 ����� ������������������������������'%������������	�������������������������������#>b{�����zT<0
���HHHTUX^[TH?;=BHHHHHHkx������������zrkkmk���������
�������������������������������������

�������������%/8BO[hmhbc[OB6-)!"%���� %' ���������`amz{�����zmda``aba`������������������������

�������������������������������moz�������������zqmm�������������������������������������zu���������������������IO[ht�����|th[ROJGFI����������������������������������������������������������))-+5BN[gmmklg[NB6,)qu}�������������vrpqY[gt��������tgb[XTYY*/6<HLQROH</.*******����������������������������������������9<HTU\ablnsb_UH?<339U^nz���������naUQOPU����������������������������������������FQ[hipstttqh[YWSOKFFY\bdit���������t[TTY 	)5?@85+��� Xdt�����������tg_ZTX��������������������������������������uu������������<<AIKII@<;69<<<<<<<<#'+/6<=</#"HHRUaakla`UHFEHHHHHHE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦©²³¶²²¦����������������������������������������E*E"E"E*E*E7E:ECEPEUEUEQEPECE7E/E*E*E*E*�<�/�#�����/�<�H�U�a�n�t�t�p�d�U�H�<�g�Z�I�E�I�Q�Z�g�s�������������������s�g�����s�g�^�a�d�m�s�����������������������H�<�>�H�U�a�j�g�a�U�H�H�H�H�H�H�H�H�H�H�����������������������������������U�U�H�H�H�I�Q�U�^�a�c�m�k�e�a�U�U�U�U�U�H�B�H�H�U�a�b�a�_�U�H�H�H�H�H�H�H�H�H�H�m�`�b�b�y��������������������������y�m�U�O�H�=�<�:�<�H�U�V�Z�V�U�U�U�U�U�U�U�U�`�`�c�k�l�n�y��������������y�p�l�`�`�`�)�#� �"�)�/�6�B�O�[�h�q�h�c�a�[�O�B�6�)àÞàèìùþüùìàààààààààà�������������������������������������a�U�X�a�m�n�n�z�|�z�s�n�a�a�a�a�a�a�a�a������������	��!�"�+�/�0�.�"��	��ÓÓÓÓ×àìù��������þùìàÓÓÓÓ�_�e�p�s�������������ÿ������������y�m�_����ۺٺ�����������������������(�-�4�6�4�,�(�������������x�s�p�h�g�g�g�s�����������������������Ŀ������������Ŀѿܿݿ�߿ݿѿͿĿĿĿļM�G�@�8�4�9�@�C�M�Y�f�g�p�s�w�t�r�f�Y�M���{�g�N�<�5�7�Y�s�����������������������������������������������������������������������������������0�2�-�.�"�	�����亽�����r�L�@�3�L�Y���ɺ���*�+�!��ֺ��׾ԾʾȾʾվ׾�����	��	�������׾׻����x�q�l�j�j�l�p�x���������������������������������żɼ����������������������������x�l�t�����������û˻ͻ̻ɻû��������������ȼ�����!�-�.�(������ּʼ������ݿ���� ����)�3�(��������������ƷƷ�����������$�.�.�0�����������=�2�0�$�+�0�=�F�I�P�J�I�=�=�=�=�=�=�=�=���
����3�Y���������������~�e�3�'��R�I�6�1�)�)�$�2�I�U�b�n�u�}Ń�x�i�b�W�R�ѿпĿ������������Ŀƿѿֿݿ��ݿѿѿ��a�=�,�%�#�)�9�H�a�m�����������������z�a������������������������������	������ù��������ùϹܹ����
�������ܹϹ�ŠřŔŌőŔŠŧŭŹ������������ŹŭŠŠčĂĈċčĚĞĦĨĳĵĳĦĚčččččč������߻�����'�)�0�2�0�'��
����*����������������*�6�C�G�M�P�K�C�6�*ŭŠŔŏŌŔŔŠŭŹ����������������Źŭ����������	��������	�����FF E�E�E�E�FFF$F/F0F&F$FFFFFFF�������������Ŀѿݿ������ݿѿĿ������������������ ���$�%�(�*�$�������𿟿������������������ĿѿѿпɿĿ¿������E�;�4�3�6�<�H�T�a�m�z�����������z�a�T�E�l�e�d�l�x�~�����������x�l�l�l�l�l�l�l�l������������������������������������������������������������������������������������²�|¦¿����� ��������ùòìäÝâìù����������������������ùĿĳĪĦğĥĳĿ���������
���������Ŀ�!���!�.�:�B�:�4�.�!�!�!�!�!�!�!�!�!�!�����y�`�V�U�d�w�����������Ľ½���������ììùý��þùìäàÛ×àììììììì��������������������������������DbDWDbDoDrD{D�D�D�D�D�D�D{DoDbDbDbDbDbDbE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� U Q K \ ! * 5 : Q ` C $ V , & K N 8 3 & � V " B 4 7 f P  g R \ " # b P < * H ! H 5 ] ! $ = % )  @ %  l B * } d b X 0 0 ^ D L M | ,    �  �  �  �  �  �  �  g  �  �  8  �  p  �  �  ,  �  o  |  �  �  �  �    �  �  ]  �  ?  Y  �  �  b    P  R  �  j  9  �    m  Q  �  4  �    m  �  n  �  \  u  M    G  �  �  D  �  �  =  $  �  +  �  �  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  �  �  �  �  z  h  S  >  %  	  �  �  �  k  '  �  �  �  b  A  Q  E  9  *      �  �  �  �  z  ]  >    �  �  �  ^    �  *  $            �  �  �  �  �  �  �  �  �  z  e  O  :  )    	  �  �  �  �  �  n  H    �  �  5  �  Y  �  1  p  �    ?  ]  i  j  f  ^  Q  <    �  �  k  
  �  �    "  )  K  �  1  �    m  �  �  �  �     �  �  �  5  �  n  �    9  �    ?  8  0  %      �  �  �  �  �  t  ]  B  %    �  �  :  J  S  [  `  b  c  _  \  V  Q  J  A  8  .  $  6  S  a  Y  Q  1  G  T  S  b  o  j  `  T  B  -    �  �  �  �  �  �  �  �  �  �  �  �    /  M  m  �  �  �  %    �  !  )  &  c  �  e  =  5  ,  #      	  �  �  �  �  �  �  �  �  �  �  x  g  V  #        +  -  '      
  �  �  �  �  �  �  r  J  *      �  �  �  �            
  �  �  �  �  �  [  0    �  �  �  �  v  >    �  �  �  �  R    �  �  �  R    �  �  S  "  �  �  �  �  �  �  �  z  a  I  :  4    �  �  X    �    �  �  �  �  �  �  �  �  �  �  �  �  �  |  v  q  k  e  `  Z  T  �  �  �  �  �  �  �  �  �  v  l  c  Y  P  G  >  4  +  "    �  �  �  �  �  �  �    w  n  e  \  S  F  8  )      �  �  �  "  ?  V  i  z  �  �  �  |  l  X  @  '    �  �  W    �  "          �  �  �  �  �  g  >    �  �  w  =    �  �  	  +  �  �  �  �  �  �  �  �  �  q  A  	  �  �  �  @  �   �  �  �  �  �  �  �  �  �  �  e  F  &    �  �  �  �  \    �  s  j  `  W  N  D  :  0  &      �  �  �  �  �  �  w  h  Y  �  �  �  �  �  �  �  �  ~  k  W  C  -    �  �  �  r  4   �  �  �  �  �  �  �  �  �  v  h  Z  L  <  -      �  �  �  '  ~  |  w  n  _  J  4      �  �  �  �  i  =    �  c  �      �  �  z  3  �  �  X  %  �  �  �  [     �  �  "  �  {   �  �  �  �  �  �  �  �  �  q  b  N  7        �   �   �   �   �   c  �  �     �  �  �  �  �  �  �  �  �  ~  @  �  �  �  �  _   �  �        �  �    E    �  �  �  �  �  �  �  e    {    �  
     3  B  L  P  Q  G  .  �  �  j    �  s    �  i  6  �  �  �  �  �  {  b  =    �  �  �  �          �  �  �  �  �  �  �  �  �  s  d  U  B  +    �  �  y  8  �  �  ^     �  [  W  L  >  ;  *    �  �  �  �  �  �  {  Y  ,  �  �  O  T  (  M  U  N  C  :  D  1    �  �  �  r  ,  �  }    �  ,    �  |  �  �  �  �  {  V  1      �  �  �  �  �  \    �  �  :  J  ?  &  %    �  �  �  �  X    �  }  !  �  Y  �  �   �  �  �  �  �  t  h  [  J  8  &    �  �  �  �  �  {  a  G  -  
�  
�  
�  
�  
�  
�  
�  
�  
l  
  	�  	E  �  p  �  8  N    �  �  W  �  �  w  \  -  �  �  }  K    �  �  �  �  [  *  �  �  �        �  �  �  �  �  �  �  �  �  �  �  s  a  O  =  ,    ^  P  C  8  +  "      �  �  �  t  8    �  u    �  +  �  �  �  �  �  �  t  d  T  @  -    	  �  �  �  �  �  ^     �  p  v  |  �  �  �  �  �  �  �  �  �  Y  �  _  �    Z  W  �  �  x  f  Q  9  "  
  �  �  �  �  g  ?    �  �  �  c    �  �  �  �  �  �  �  �  u  k  U  <  $    �  �  �  �  �  �  t  n  n  g  Z  D  -    �  �  �  �  �  �  �  Y  &  �  �  H  �  U  H  E  ?  <  ?  C  B  <  3  "    �  �  �  ;  �  �  n    =  `  {  �  �  �  �  �  �  n  L  !  �  �  \    �  D  �  �  I  M  J  F  @  6  #    �  �  �  f  7  	  �  �  �  _  =  ,  C  ;  2  '      �  �  �  �  �  q  \  B     �  �  ?  �  N  \  p  ~  ~  s  f  V  A  (    �  �  4  �  �    �    /  �  y  ]  B  '  
  �  �  �  �  �  {  j  l  n  |  �  �  �  �  �  .    �  �  �  �  [  #  �  �  �  p  1  �  �  x  L    �  z  K  f  �  �  �  {  j  S  5    �  �  E  �  ~  �  P  f  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  i  U  B  )    �  �  �  e  4    �  �  a  )  L  5    �  �  �  �  �  |  e  L  <  "     �  �  x  7  �  �  �  �  n  b  �  �  �  �  �  ^  .    �  f    �  e  �  ]    C  	  	!  	   	  	  �  �  �  p  =    �  `  �  &  c  �  �  �  �  �  �  l  9    �  �  n  B    �  �  8  �  s    �  R  j  t  \  E  0          :  b  e  B     �  �  �  �  b  8    �  �  �  �  x  j  W  @  %    �  �  �    ^  :    �  n  �  ,    �  �  �  �  �  {  [  2    �  �  Q    �  j    �  �  j  ^  R  F  :  .  "      �  �  �  �  �  �  �  �  s  b  Q  �  �  x  P  &  �  �  �  ~  c  `  g  u  L    �  �  �  R    	_  	+  �  �  �  e  (  �  �  U  �  �  Z  
  �  ^    �  
  �