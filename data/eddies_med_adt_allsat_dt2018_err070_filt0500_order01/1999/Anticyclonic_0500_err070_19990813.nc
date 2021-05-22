CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?ӥ�S���       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�8�   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       =�h       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @Ffffffg     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
<    max       @v�33334     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @���           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >��D       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2 v       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B1��       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?
I�   max       C��&       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?;   max       C���       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         4       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�8�   max       PI8�       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�PH   max       ?�ᰉ�'S       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >,1       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @Ffffffg     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
<    max       @v�33334     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q            �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @�            V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?ye+��a   max       ?��f�A�     P  X�               9      E                  4         (         9                        	   U   9     3   	             3   .   
   	      !            .                        %         5               �            OI��M�TN���N6�@O� JN鑻Pc��N��gN�wM�EN�OyozP4N���O|G[Ox�O:'O�V,PB�N�M�8�OԷN^�xNx<�OP��N5ZgO9��P��O���N�lP~y!N�|)N��N�>�O��<P� O�^�N���N��N�q�OPb�N��O)��OD�fO�^�M�p�N��XN���O��\O9eO���N��OŻUO_�O	O�.TO`3�N]�aN��dO)�O�TN�O:�RN�|TNJ2F�C����
�o�ě�;D��;ě�;�`B<o<o<#�
<49X<D��<T��<T��<u<u<u<�C�<�t�<�t�<�t�<��
<�9X<�j<�j<�j<�j<���<�/<�h<�<��=o=+=+=C�=t�=��=,1=0 �=49X=8Q�=8Q�=<j=D��=T��=T��=Y�=]/=aG�=aG�=e`B=ix�=ix�=q��=u=u=y�#=�o=��=�C�=��=ȴ9=ȴ9=�h#/<DHOQSHB</#!RU\bnornbURRRRRRRRRRnnnnnmn{|����}{nnnnn�����������������������
#/<DHD<#
�����������������YZ]gt�����������tf\Y��������������������.01<@HGIUbiiebUNI10.��

����������������������������������������������������5CF>5)����������

�����#/0HINPPLH</##/<HV``YUHA<-#�����������������������������������������������/43/+#
����OITUUadhaUOOOOOOOOOO"#&010%#"
#/995/#
����������������������������������������-36BJN[hjsprlh[OB76-z�����������zzzzzzzzJFFHMOW\ahu}�}{vh\OJ�����5MZ_]R>:)���������
$(.+#
���a^chqt�������ythaaaa-*-9Ngt���������gN8-��������������������--+1569BOPSTONFB:6--�������������������������������������������)BJ[SN2)����������������������������������������}���������������������������������������������]\\]`amz�������zmba]$).-)'����������������������z{{~��������������������������������������������������������������������������z�������������zzzzzzlmkfdecnz��������znl������

�������[`kt�������������qg[!')5BMNRNB5)!!!!!!!!5/1=Ljm������zmaTH;5������ ����
	)069;:766)
������6?BLGB.�QPY[pt�����zzthefc[Q��������������������2.**6:BFNKGB<6222222;;AEHU\ahpz}znaUHF<;����������

���1,158BHDB51111111111�������
���')*55975)��������������������EEE$E.E/E#EEEED�D�D�D�D�D�D�D�D�E�������������������������лܻ������������ܻлʻλлллл_�l�x�{�x�q�l�_�[�T�_�_�_�_�_�_�_�_�_�_�m�y���������������y�`�T�J�8�5�;�?�T�`�m�����������û̻лٻлû������������������/�H�T�_�a�]�O�H�;�/�����������������/�"�/�;�C�>�;�;�/�$�"���"�"�"�"�"�"�"�"�f�r���������������������|�u�r�q�f�[�f�M�R�Z�b�f�s�v�s�f�Z�W�M�L�H�M�M�M�M�M�M�����������������������������������������z�����������������������z�r�f�b�e�m�o�z�/�<�a�ÇÄ�m�[�H�<�#���!���
�&�$�/�;�H�T�U�\�a�b�a�T�H�;�7�/�.�/�5�;�;�;�;������������������������������������������������������������������������޺������!�(�*�!��������������"�&�-�1�1�,�"��	����׾ξо׾�����5�B�[�b�h�{�o�m�e�N�)�#�������������5���)�)�)����������������S�U�_�d�_�W�S�P�F�F�F�Q�S�S�S�S�S�S�S�S�g�k�s�s�g�b�[�A�5�(���(�*�5�A�Z�]�^�g�M�Z�]�f�i�s�{�s�f�Z�W�M�J�J�M�M�M�M�M�M²¿������������¿²¦¢¦«²²²²²²��#�&�1�<�H�V�W�U�<�/�#������
���<�>�H�R�H�H�<�/�-�+�/�9�<�<�<�<�<�<�<�<�	��"�.�;�B�G�K�G�E�B�;�.�"��	����� �	���������#�'�����Ƴƚ�u�C����*�u�������׾���������ʾ������������������'�3�@�E�L�P�M�L�E�@�:�3�3�'�%��'�'�'�'����)�B�R�]�]�V�K�6�)�����������������������������������������x�������������������������������ܻٻܻ߻��4�8�A�F�G�A�4�)�(�'�������(�2�4�4����#�'� ����ݽĽ������������н��������������������������������������������T�a�m�x�~�����z�r�a�T�;�&�"����"�;�T��������������������������������������������"�(�+�(�(���������������a�n�t�v�t�z�{�z�n�e�a�]�X�W�U�^�a�a�a�aŔŠŭŹ����������ſŹŭŠŔňŁŁŇŉŔ��������� ����߿ݿԿݿ߿������ÇÓàìõùü��ùöìàÓÇ�}�z�z�}ÄÇ��������������������������������ҽl�y���������������y�`�S�A�>�A�G�S�`�i�lF=FJFVF\FcFdFcFVFMFJF=F2F=F=F=F=F=F=F=F=�ܹ������������ܹӹϹǹϹ׹ܹܹܹܿ������ĿѿҿڿѿĿ��������������������������ֺ��������������պɺ��������H�U�a�d�j�n�n�i�a�U�<�5�/�$�"�#�,�/�<�H��(�4�;�I�Q�V�V�I�A�4�(������������!�"���������������čĚĦĳĿ����������ĳĦĚČ�~�{�{�}ąč�;�G�T�`�e�k�c�U�G�;�.�"��	��	��"�.�;�~�������������������������~�~�r�m�r�u�~������������������������r�f�d�M�G�K�W��f�r��������������r�a�Y�@�3�0�4�@�M�Y�f���'�(�1�(�������������������!�%�!�����������������������(�4�A�M�S�T�M�I�A�4�(��������&�(DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DtDjDhDo�`�m�y�{�y�t�m�`�]�X�`�`�`�`�`�`�`�`�`�`�
��#�0�:�<�B�<�9�0�#���
���������
����������������������������������������E7ECEPEUEVEPECE7E6E2E7E7E7E7E7E7E7E7E7E7 & e q -  B . > p � 3 2 C 9  ( : & + J O Z l O I J  U R <  @ + 5 V 1 I ` 4 ` '   , # p ? i K ! B _ K K 1 < W : R K  4  , 8    �  C  �  F  o    �  �  N  �        �  �  �  ,  n  O      �  �  �  �  k  �  7    �  �  �    �  �  �  �  �  �  �  �  �  b  �  s  K  �  �  Z  �  �  �  �  �  F  6  
  k  �  �  y  '  �  �  k�t���t��o�o=u<�C�=���<#�
<e`B<T��<e`B=+=��<u=+=]/<�/=\)=���<�j<�1<��<�h=�P=\)<�`B=o=�S�=�1=49X>��D=#�
=H�9=#�
=��=�{=��=@�=L��=]/=��-=D��=�hs=}�=�v�=ix�=q��=u=��=���=�1=y�#=�v�=�-=��-=�`B=� �=��-=���=��T>o��=��w=�h=�;d>oB)�B'��B(y�B�B��B#@oB
ͱB��B&��B$!&BrLBi(BH�B��BD�B��B �B �dB�@B�B%9�B߾B�PBIB�<B5/B2 vBG9B�B_B	:�Bq�B�B �B"HxB��B23B�OB�PBi,A�4�BDB!��B ��B,]�B��B�2B
�MB��B��B�BXVA���B�/B�B6�B?�B��BB��BƿBpB�B��BU�B8KB(0B(�)B�QB�B#<�B
�vB��B'?�B$A2BH�BE�B�.B�/BK/B�GB ��B �B��B�\B%B*B88B�B߃B>BD�B1��B�5BHxB��B	=�BG B?�B�jB"I�B��B?�BDB�BJ�A��B8 B!��B ��B,@�B��B��B
�/B��BB�Bz�BB�A�B�B�BF�B@B��B9/B�aB�B@�B�8B��B�C�]�A-@���@���Ajފ@�chA�m�A�aA@�6�A@T�Ar�vA�x�A���A���A�?hA�R�@ZmAY�RA�?UA�N#@�E�A�I5A?�A���A�"�AÄ�A^�9B�AQj?�iA�p�A���@���A8[�A,[�A�՘A�O!@|-A��A�(A��A��A�#A��A��C��&?
I�Aw�b@B��AĐ�A7�A3@DA�;�Ab�E@��@�ɚ@�FA�D�@\�QA8
�C��Aj�NA�7�A�GC��}C�XgA-@�փ@�L�Aj��@�ltA�'A��@��A>��As4A�A�Z�A���A�$A�5�@\�AZR�A�|[A�~a@�&A��9A>��A��A��A�A^2�BB.AR�
?��A�~fA��&@���A8��A,�A��eA��@�5A�GA�{-A�A�1�A��A�	aA�C���?;Aw2@A�4A�-�A8��A3�A��Aa)�@P0@��@��A���@[�.A8�fC���Aj�A�˓A�}_C��'               :      F                  5         (         :                        
   V   :     4   
         !   4   /   
   	      "            .                        %         5               �                                 3                  -                  -                           E   !      1            !   )   !                                                         #                                                !                  %                                             1         !                                                                                                      O��M�TN���N6�@OL��N�\	O�;�N��gN�wM�EN�O Y`O� 9N���O!�FN�7�O:'O�V,O��N�M�8�N�@!N^�xNx<�OP��N5ZgO9��PI8�O��N�lO�YN�|)N���N\dO0�YO�2OJ�N���N��M�iOP�N��O �OD�fOp%HM�p�N��XN���O�H�O9eO���N��OŻUO_�O	O���O`3�N]�aN��dO)�OO׃N�O)��N�|TNJ2F  �  ?  E  �  �  I  �   �  �  Q  �  v    /  R  �  O  .  �  k  m  �  s  �  �  �  �  �  �  �  �  :          �  @  :    �  Q  <  �  �  �  �    �  �  �  H      �  z  6  #  3  G  �  �  &  �  s��`B���
�o�ě�<�1<o=t�<o<o<#�
<49X<���<�9X<T��<��
<�h<u<�C�=�P<�t�<�t�<�1<�9X<�j<�j<�j<�j=P�`=�w<�h>,1<��=t�=\)=49X=@�=D��=��=,1=L��=H�9=8Q�=<j=<j=m�h=T��=T��=Y�=aG�=aG�=q��=e`B=ix�=ix�=q��=��P=u=y�#=�o=��=�=��=���=ȴ9=�h#/:<DGE<5/,#RU\bnornbURRRRRRRRRRnnnnnmn{|����}{nnnnn��������������������
#/3<?@<9/&#
�����������������gitx������������tlhg��������������������.01<@HGIUbiiebUNI10.��

����������������������������������������������������5<?5)���������

�����#/;<HHJKHE</#"!!#-/8<HLSRHG</+#""�����������������������������������������������!!
�����OITUUadhaUOOOOOOOOOO"#&010%#"#,/4884/#����������������������������������������-36BJN[hjsprlh[OB76-z�����������zzzzzzzzJFFHMOW\ahu}�}{vh\OJ����5BPTRJ5)���������
!!
����a^chqt�������ythaaaa?>@FN[gt������tg[NC?��������������������10056;BNOPOJCBA61111�����������������������������������������������8@A<5)������������������������������������}���������������������������������������������``abhmz{�������zmja`$).-)'����������������������z{{~��������������������������������������������������������������������������z�������������zzzzzzmkgefenz��������~znm������

�������grt��������������thg!')5BMNRNB5)!!!!!!!!5/1=Ljm������zmaTH;5������ ����
	)069;:766)
����'1564-)���QPY[pt�����zzthefc[Q��������������������2.**6:BFNKGB<6222222;;AEHU\ahpz}znaUHF<;������� 

������1,158BHDB51111111111����������')*55975)��������������������EEEE'E(EEEED�D�D�D�D�D�D�EE	EE�������������������������лܻ������������ܻлʻλлллл_�l�x�{�x�q�l�_�[�T�_�_�_�_�_�_�_�_�_�_�`�m�y�������������y�m�`�W�T�L�G�E�I�X�`�����������ûɻϻû����������������������"�/�D�H�K�K�H�;�/�"�	�������������	��"�"�/�;�C�>�;�;�/�$�"���"�"�"�"�"�"�"�"�f�r���������������������|�u�r�q�f�[�f�M�R�Z�b�f�s�v�s�f�Z�W�M�L�H�M�M�M�M�M�M�����������������������������������������z�����������������z�u�m�m�m�m�n�z�z�z�z�<�H�b�zÂ��g�V�H�<�+�+�)�&�$�!�#��/�<�;�H�T�U�\�a�b�a�T�H�;�7�/�.�/�5�;�;�;�;������������������������������������������������������������������������������!�(�*�!��������������"�&�-�1�1�,�"��	����׾ξо׾�����)�5�B�N�a�c�`�[�N�B�5�)��	����� ���)���)�)�)����������������S�U�_�d�_�W�S�P�F�F�F�Q�S�S�S�S�S�S�S�S�5�A�N�R�Z�_�Z�Y�N�A�5�(� ��(�1�5�5�5�5�M�Z�]�f�i�s�{�s�f�Z�W�M�J�J�M�M�M�M�M�M²¿������������¿²¦¢¦«²²²²²²��#�&�1�<�H�V�W�U�<�/�#������
���<�>�H�R�H�H�<�/�-�+�/�9�<�<�<�<�<�<�<�<�	��"�.�;�B�G�K�G�E�B�;�.�"��	����� �	����������� ������ƳƚƁ�]�M�Q�e�uƚƧ�����ʾ����������ʾ������������������'�3�@�E�L�P�M�L�E�@�:�3�3�'�%��'�'�'�'����)�3�=�B�A�6�)�����������������������������������������x������������������������ �����ܻܻܻ�����(�4�A�D�D�A�4�(�����(�(�(�(�(�(�(�(�ݽ�����������ݽнĽ��������Ľ̽������������������������������������������;�H�T�a�m�q�v�t�m�m�a�T�H�;�9�.�,�/�4�;��������������������������������������������"�(�+�(�(���������������n�n�p�r�n�b�a�a�^�_�a�m�n�n�n�n�n�n�n�nŠŭŲŹ��������ŻŹŭŠŔŋŇńņŇŔŠ��������� ����߿ݿԿݿ߿������ÇÓàìòùü��ùõìàÓÇ�~�z�~ÆÇÇ��������������������������������ҽy�����������������y�l�`�S�J�D�G�S�`�l�yF=FJFVF\FcFdFcFVFMFJF=F2F=F=F=F=F=F=F=F=�ܹ������������ܹӹϹǹϹ׹ܹܹܹܿ������ĿѿҿڿѿĿ������������������������ֺ������
�������ֺɺ����������H�U�a�d�j�n�n�i�a�U�<�5�/�$�"�#�,�/�<�H�(�4�6�E�N�R�Q�E�A�4�(���
�
�����(����!�"���������������čĚĦĳĿ����������ĳĦĚČ�~�{�{�}ąč�;�G�T�`�e�k�c�U�G�;�.�"��	��	��"�.�;�~�������������������������~�~�r�m�r�u�~��������������������r�f�Y�V�P�T�f�r����f�r��������������r�a�Y�@�3�0�4�@�M�Y�f���'�(�1�(�������������������!�%�!�����������������������(�4�A�M�S�T�M�I�A�4�(��������&�(D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDrDvD{D�D��`�m�y�{�y�t�m�`�]�X�`�`�`�`�`�`�`�`�`�`��#�0�8�<�A�>�<�7�0�#��
��������
������������������������������������������E7ECEPEUEVEPECE7E6E2E7E7E7E7E7E7E7E7E7E7 # e q -  B ) > p � 3 4 B 9   : & % J O , l O I J  = I <  @ - ) K  ' ` 4 � $   , $ p ? i G ! 8 _ K K 1 ; W : R K  4  , 8    #  C  �  F  �  �  �  �  N  �    *    �  W    ,  n  �      �  �  �  �  k  �  q  D  �  �  �  �  `  �  {  �  �  �  {  T  �  S  �  �  K  �  �  6  �    �  �  �  F  F  
  k  �  �  �  '  g  �  k  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  k  |  �  �  �  �  �  �  l  K  (  �  �  �  /  �  p    �  G  ?  >  =  <  ;  :  9  8  7  6  7  9  <  >  A  C  F  H  K  M  E  C  @  >  :  6  1  '      �  �  �  �  �  �  �  x  b  K  �  �  �  �  �    s  f  Z  M  @  2  $       �   �   �   �   �  m  �  �  �  �  �  �  �  �  �    <  �  e  �  A  �  �  �  �  ;  B  G  H  F  A  6  +      �  �  �  �  �  c  5  �  �  �  �    :  W  w  �  �  �  �  �  �  c    �  ]  �  n  �  5  �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  l  X  A  )     �   �   �  Q  N  L  I  F  C  @  3        �   �   �   �   �   �      i   S   >  �  �  �  �  �  �  �  �  �  w  m  b  X  M  >  0  !       �  �  �    ;  T  f  r  u  r  m  e  Y  G  /  	  �  s    f   �  �  �  �      �  �  �  u  M  )  1  	  �  �    H  ~    �  /  -  +  (  &  $  "              #  &  )  ,  /  2  5  9  !  9  D  J  O  Q  Q  M  B  0      �  �  �  |  N    �  ;  �  :  g  �  �  �  �  �  �  �  �  �  Q    �  o  �  Z  �  �  O  A  3  %      �  �  �  �  �  �  �  �    L    �  �  c  .  '    	  �  �  �  �  �  �  �  z  k  X  B  (    �  �  Z  �  �  9  w  �  �  �  �  �  �  c  (  �  p    �     P  [   �  k  g  c  `  Z  K  =  .    �  �  �  �  �  a  >     �   �   �  m  g  a  \  V  P  K  C  :  1  (           �  �  �  �  �  t  �  �  t  _  N  >  .      �  �  �  �  �  �  �  �    �  s  ^  I  3      �  �  �  �  �  l  K  )  �  �  {  7  �  �  �  �  �  �  �  �  �  �  �  �  |  i  U  ?  '    �  �  �    �  �  �  �  �  h  R  J  J  ^  h  j  `  T  G  8  (        �  �  �  �  �  �  �  �  w  j  ]  O  K  L  M  N  L  I  F  D  �  �  �  �  �  �  �  �  �  �  �  �  u  e  M  2     �   �   x  Y  �  �  �  �  �  �  �  �  �  x  a  ?    �  _  �    �  �  g  �  �  �  �  �  �  �  �  �  q  '  �  m    �  "  �    �  �  �  �  �  �  �  �  �  �  z  c  L  5      �  �  �  F  �  �  �  V  s  z  ?  �  W  �  �  a  �  O  l    i  !  a  e  W  :  2  *        �  �  �  �  �  �  �  ~  h  P  2    �  �  �  �          
  �  �  �  �  �  �  �  T    �  �  ^    �  �  �  �  �          �  �  �  �  �  �  �  �  �  |  U  x  �  �  �  �      
  �  �  �  �  T    �  �  R  "  �  �  T  x  �  �  �    �  �  �  �  �  �  i  :  �  �  @  �  �  S  3  j  �  �  �  �  �  �  �  �  Y    �  `  �  }  �        @  %  
  �  �  �  �  ~  d  K  3    �  �  �  �  �  �  p  U  :  2  *       
  �  �  �  �  �  �  �  �  �  u  a  P  E  :    /  @  O  _  o  }  �  �  �  �  �  	  C  k  B    �    �  �  �  �  �  �  �  �  �  v  \  D  $  �  �  ;  �  �        Q  N  J  G  D  A  >  9  3  .  (  "          �  �  �  �  3  ;  4  %    �  �  �  �  W    �  w  #  �  �  O    �  �  �  �  �  �  �  �  �  �  o  W  ;    �  �  �  n  3  �  �  �  �  �  �  �  �  �  �  �  x  J    �  �  Z  8    �  1  �    �  �  �  �  �  }  m  \  I  2      �  �  �  �  �  �  s  `  �  �  �  �  �  �  �  }  u  ]  D  +      �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  |  6  �  &  �  �  �  �  �  �  �  �  _  3    �  �  �  v  N    �  �     �  3  �  �  �  Z  <    �  �    F  
  �  �  J  �  �  q  1  �  �   �  �  �  �  �  �  x  f  Q  :       �  �  h    �  _    �  �  H  @  8  0  '        �  �  �  �  �  r  F     �   �   �   �      �  �  �  �  e  =    �  �  W    �  b  �  �  �    L        �  �  �  �  �  �  k  T  9    �  �  #  �    c  �  �  �  �  �  �  �  �  x  b  L  5    �  �  �  �  {  Y  6    �  (  F  _  r  z  r  b  H  &  �  �  v    �  <  �  �  +    6    �  �  �  �  �  g  O  <  ;  C  J  /  �  <  �  L  �  @  #    �  �  �  �    a  8    �  �  2  �  �  1  �  y     �  3  )      �  �  �  �  m  H  '      �  �  �  �  �  k  L  G  2      �  �  �  �  �  �  �  g  K  4  #  )  >  Y  w  �  �  |  �  n  �  �  �  �  �  B  �    <  g  \  �  "  	  
�  �  �  �  �  v  f  T  B  0      �  �  �  �  �  i  H  '     �    %        �  �  �  �  �  x  P  (  �  �  �  s  <  �  �  �  �  r  b  P  ;  $    �  �  �  �  q  .  �  �  F  �  �  Z  s  G    �  �  �  Y  $  �  �  �  �  j  E    �  �  �  �  