CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�Q��R      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��4   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?p��
=q   max       @F�\)     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vt��
=p     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @��@          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >�1'      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Mi   max       B/ܝ      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�p�   max       B/�d      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��p   max       C��      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�C|   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��4   max       P�.�      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���N;�6   max       ?�ۋ�q�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       >O�      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��\(��   max       @F�\)     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vt          	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�e�          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B<   max         B<      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?���>BZ�     �  V�                           
               *                           P   	   3               /   @   !      $               /      	            B      	   *   1   	   %            B   %                  �   O\�Ne��N�s�O�ZO6�Oµ�N55AN��2N��N��N�V-N	�N�OoP<�O��,O�H�OkAEOtNވ�N$��N9CBOV�WPj aO��PT?1N튤NC<UN@��N�;P;�sP�YOV7N��~O���NN��Nӆ�N�&P��O*zzNK�\O���OQ��NB�KP��M��4N�/�O֠8P(N��dO�dNX!Nk��O��P	��O˙
NS�iN���N��EN"fNU؏O�]�Nd����
�#�
���
�o��o��o��o%   %   :�o;�o;ě�<t�<#�
<#�
<#�
<49X<49X<49X<D��<D��<T��<u<�C�<�C�<�t�<���<��
<��
<�1<�1<�9X<�9X<���<�`B<�`B<�<�<��<��<��=+=+=\)=t�=�P=�w=�w='�=,1=D��=D��=H�9=]/=aG�=ix�=q��=q��=u=�o=�hs=���=�����������������������#-030+#!ECENT[egmpiga[SNEEEE��������������������)5798855))1*)05BN[gpwurg[NGB51|�����������||||||||[Y^ammz��|zoma[[[[[[�������������������� �%),+)      ����������������������������������������&)5BJNPONB5)))&&&&&&�����
$4=QVXQ<#
���SS[������������th`[Sedht�������������theIIJHTYamz������zmaTI��������������������������������[anz���znba[[[[[[[[|z����������||||||||�������������������������)B`fg^5)��������'(���������������
/LB/
������������������������406BORYOB64444444444^ahlrtx~tsh^^^^^^^^��������������������_cn��������������kb_����)6BKNMF@6)��-(*/<HMU`lnca\UJH</-������������������������#)4:A>4)���������������������#'/<HHLJHC</$#adgqt��������tngaaaa�����������������������G[t����tgB5�}zz|}��������������}����

������������������
!)*'#
�����#/4<@BA@><1/#�������������������������)6BLN<.) ���)&"*/00676*))))))))�����������������������*120)	����������/6<972$���!#-033230(#
LT^mz��������zmeaTHL��������������������13<HU_`UH<1111111111������������������������������������)>BHNTVUXWSB5(������������������������������������� 	

�������������������������
#$#""
���������
 
�������������������������������������������Ż�����ƻ-�:�F�S�U�Y�S�L�F�A�:�/�-�(�-�-�-�-�-�-���������������������������������������ѹ���������ܹϹǹ������������������Ϲ�����	����	�����׾־ʾɾ̾׾���"�;�T�m�s�z�{�}�|�y�T�G�;�,� �����"��������������������������������������������������������������������������������������������~���������������
��	�����������������������������������	�������������������������������������������������������������������������������������������������������������	�"�;�T�a�h�a�T�;�"��	���������������	������������������f�]�a�Z�S�\�[�h�s����������(�0�4�;�4�(����нɽȽѽݽ���;�H�T�a�l�m�u�q�q�q�w�m�a�O�H�<�6�6�5�;������)�2�1�)�������������������������������������������������������������<�E�@�<�<�<�;�/�,�+�/�/�<�<�<�<�<�<�<�<�L�Y�a�e�n�n�e�^�Y�M�L�D�L�L�L�L�L�L�L�L�s��������������������������s�m�l�e�l�s����&�/�2�2�1�(���꿩�������ƿڿ����f�s��w�v�v�s�o�f�Z�N�O�S�T�Z�]�f�f�f�f����(�A�g���������s�Z�=������ؿտ׿��(�4�8�7�4�2�(�'���
���������z�����������z�n�r�y�z�z�z�z�z�z�z�z�z�z�-�:�B�F�S�U�S�F�:�.�-�'�-�-�-�-�-�-�-�-�/�<�B�G�>�<�4�/�%�$�%�*�/�/�/�/�/�/�/�/�������������������s�Z�J�:�7�A�g�s�������������	�/�;�L�L�B�/�"�	������������������������������������������������޿���"�)�*�"��	����	�	��������h�tāčěĦĩĲĦĢčā�t�h�[�T�M�N�X�h�[�_�`�[�R�N�D�B�@�B�N�U�[�[�[�[�[�[�[�[E�E�E�E�FF	E�E�E�E�E�E�E�E�E�E�E�E�E�E��[�h�l�t�v�y�t�o�h�[�O�J�L�N�O�Q�[�[�[�[�B�G�C�B�:�6�)�&�(�)�6�7�B�B�B�B�B�B�B�B�	�"�;�Y�[�H�5�)�	���������������������	�g�s�����������������������g�Z�T�R�Z�_�g�a�n�t�t�s�n�a�_�X�W�a�a�a�a�a�a�a�a�a�a�������������������y�m�T�I�M�W�b�m�}���������"�.�8�9�.�"��	������׾Ծؾ��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٺ������ɺٺ�ܺԺ�պ����~�e�S�Y�_�c�r���l�y�����������y�o�l�k�l�l�l�l�l�l�l�l�l�<�H�T�U�V�U�N�H�>�<�7�0�/�-�'�/�1�2�<�<���#�!��
������²¡¦²¹�������
��0�<�I�U�_�b�\�U�I�0�#���������������#�0��(�4�A�T�f�k�s����s�f�Z�M�4�,�(���ĦĭĳĸĹĶĮĦĚčā�~�y�y�{āĄčĚĦ�y�������������|�y�u�y�y�y�y�y�y�y�y�y�y��� �!�����������������4�A�M�Z�_�e�`�Z�M�A�4�(�����(�-�4�4��������������������f�]�S�P�Q�X�f���������2�I�U�b�f�V�=�0������������������l�x�����������������x�m�l�g�l�l�l�l�l�làìù��üùìãàÔÓËÇÁÁÇÓÞàà�6�C�I�O�\�h�q�h�\�X�O�C�=�6�2�5�6�6�6�6������	��������������������n�y�{ŀ�}�{�r�n�b�U�T�U�b�e�n�n�n�n�n�nDoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DpDfDbDo�4�;�@�F�F�@�4�'��!�'�.�4�4�4�4�4�4�4�4 K ; P 1 : ' 5 Z ) < Y w 1 < Q ? V < Y ~ a @ ? n P 9 _ ] b @ H . S # [ . 2 f U P B r G ( " � [ V . � B : j 5 \ p \ n Y + : ( ,    M  ~  �  �  P  �  ]  �  �  �  �  u  �  ?  m  0  �  [    �  o  �     y  �    k  |  �  _  �  �  �  *  D    �  ]  �  �  n  �  �  X  �  D  �  -  �  �  	  .  �  4  �  �  �  �  �  6  s    w�t���`B;ě�<���<49X<���;o;o<t�<o<o<T��<�1=P�`<�j<���<���<�j<���<��
<��
<��
=\<���=�7L<�j<�9X<���<��=�C�=�{=aG�<���=y�#=+=q��=,1=t�=��-=@�=#�
=]/=]/=<j=��`=#�
=D��=��T=�E�=L��=�{=T��=]/=��-=�=��=�\)=��
=�7L=�7L=���>�1'=�x�B�B%�B��Bu�B��BD�B98A�w�B)��B��B�+B��B�(B�eBg�B�OA�MiBq�B!o*B��B"�B$xB�2B,tB�RB��BRB�B�BuqB��B�B"�@B-B�
Bq�B
�B��B�EB
��B"�B��B	�B�{B�SB/ܝB6�B�2B��B$�A�j�B,lB*�B�EB�FBL�BVsB"utB�B=�B5TB��B_[B�B%�B� B~B@BJRBR|A���B)��B5�B�zB��BB�B��BB�B�A�p�B?�B"4DB@HB?�BAB��B�9B��B�B�B.�BѾBFuB4�B�-B"�$B8�B�>B�)B	�CBٛB�UB
��B?zB�B:�B��B@�B/�dB@�B�B|�B%CGA���B,R�B;DB��B�}B?�B@JB"?�B�,BO�B:7B>�BE�A��@�Y�A�s$>��pAW�Ae�?dNA��@�W�A���B�VA�ҝAЁA��	AF�A/êA�n@A�l�@�,A���?��AG��A���A@ҫA��MA5c�A��@}�|A£�A��>A��$A�6lA]OA�A��gC��Aڽ�AפeA��2A�$�A���Am$�AZvC�pV@�A+/A��A�auA��lA:�9A��AU|A3w�A;�$@�	�B�h@�\�A˗�B.�A1D�A�M�C�֌@��pA��@���A���>�C|AV��Ac�?k�A� @�H�A��B��A�V�AЀA�GAGJ�A/(A��&A���@��A��?��AG1�A��A@�SA��A4��A�o@}+�A�A�zA�1A�|�A\�A�m�A�]�C���Aڏ>A�~@A�KA��WA�|-Am�-AZ�2C�w�@#�A=MA�xBA�}zA�R�A<5�A�~�A�A3��A<��@� B��@���AˁB ��A1�A�|�C��3@�(>                           
               *                  	   	      Q   	   3               0   A   "      %                /      
            C      
   +   1   	   &            B   &               	   �                                             -                           /      3               /   )                        9                  +         %   '                  )   )                  #                                             !                                                '                           7                  %            '                  #                        O\�Ne��N�7�O�5qO�eO��bN55AN��2N�;ZN6�MN�V-N	�NǀO�%lN��<O+�WON��FNވ�N$��N9CBOV�WO̕O��O���N튤NC<UN@��N�;P��O�4�N��N��~OA�NN���Nӆ�N�&P�.�O	=�N)��O.�dOE��NB�KP"�M��4Nvo�O��oPg�N�
�OW�NX!Nk��N��O싎O�D$NS�iN���N��EN"fNU؏O��Nd�  ]  �  �    �  f  �  p  �  �  $    `  U  _      �  �  �  4  �  �  �  �  �  �  �  [  �  I  (  J  �      �  Y    \  }  �  0    {    �  f  �  �  �  !    �  

  v  =  �    _  �    ����
�#�
�D��;D��%   :�o��o%   :�o;D��;�o;ě�<e`B<�1<�o<e`B<u<D��<49X<D��<D��<T��=8Q�<�C�=�P<�t�<���<��
<��
<�=#�
=�P<�9X=C�<�`B=t�<�<�=o=+=o=��=C�=\)=49X=�P=#�
=H�9=49X=0 �=T��=D��=H�9=ix�=}�=��=q��=q��=u=�o=�hs>O�=�����������������������#-030+#!GIN[bgkng[VNGGGGGGGG��������������������)4576753)0,+35BN[ioqvtpg[NB60|�����������||||||||[Y^ammz��|zoma[[[[[[��������������������(����������������������������������������5.35BCIEB55555555555������
$09@D?/
���cb^hmty��������tnhccghot|�����������{tigSOQTTamz����{znmaTSS����������������������������������[anz���znba[[[[[[[[|z����������||||||||�����������������������)5BENQSQLB5)�����'(���������������
#.462#
������������������������406BORYOB64444444444^ahlrtx~tsh^^^^^^^^��������������������hinz��������������qh���)6;ADD=6)�<1/4<HKUWUTH<<<<<<<<����������������������
)/4585)���������������������!##-/<DHF<9/.#!!!!!!adgqt��������tngaaaa������������������������)F[t����gB5�~||~��������������~~����



����������������
#%&##
���� #/3<?AA@><0/#��������������������������)6BGH9)���)&"*/00676*))))))))�������������������������%)+,)#�������)19750 ���
#,01/+'#
TRTdmz��������zmhaYT��������������������13<HU_`UH<1111111111���������������������������
	��������$)5>BKNTSNB5)������������������������������������� 	

�������������������������
#$#""
��������

��������������������������������������������Ż�����ƻ-�:�F�S�U�Y�S�L�F�A�:�/�-�(�-�-�-�-�-�-���������������������������������������ѹϹܹ���������ܹϹù������������ùϾ�����	���	������ݾ׾̾ξ׾����.�;�T�m�o�w�x�{�y�m�`�T�G�;�.�"���"�.�������������������������������������������������������������������������������������������������������������������������������������������	��������������������������������������������������������������������������������������������������������������"�/�H�S�Z�^�Z�T�;�/�"��	���������
������������������������u�s�s�r�s�s���������'�������ݽнн׽ݽ����;�H�T�a�a�h�j�j�f�a�Y�T�H�E�;�;�:�;�;�;����(�)�0�*�)������������������������������������������������������������<�E�@�<�<�<�;�/�,�+�/�/�<�<�<�<�<�<�<�<�L�Y�a�e�n�n�e�^�Y�M�L�D�L�L�L�L�L�L�L�L�s��������������������������s�m�l�e�l�s������!�#�!���������տſ����ѿ��f�s��w�v�v�s�o�f�Z�N�O�S�T�Z�]�f�f�f�f����(�6�A�O�T�O�9�(����������������(�4�8�7�4�2�(�'���
���������z�����������z�n�r�y�z�z�z�z�z�z�z�z�z�z�-�:�B�F�S�U�S�F�:�.�-�'�-�-�-�-�-�-�-�-�/�<�B�G�>�<�4�/�%�$�%�*�/�/�/�/�/�/�/�/���������������������g�Z�S�D�A�N�g�s�����������	�"�,�7�9�5�/�"��	�������������������������������������������������޿���"�)�*�"��	����	�	��������tāčďĚğĢęčā�}�t�h�[�V�Z�[�h�i�t�[�_�`�[�R�N�D�B�@�B�N�U�[�[�[�[�[�[�[�[E�E�E�E�FFE�E�E�E�E�E�E�E�E�E�E�E�E�E��[�h�l�t�v�y�t�o�h�[�O�J�L�N�O�Q�[�[�[�[�B�G�C�B�:�6�)�&�(�)�6�7�B�B�B�B�B�B�B�B�	�"�;�T�Z�V�H�4�(�	�������������������	�g�s�������������������s�g�Z�W�U�Z�c�g�g�a�n�r�s�p�n�a�a�Y�Y�a�a�a�a�a�a�a�a�a�a���������������}�y�m�`�^�T�T�\�`�f�m�x������"�.�6�7�.� ��	������׾վھ���E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٺ~�������ɺѺۺкͺӺʺ����~�c�_�a�i�r�~�l�y�����������y�o�l�k�l�l�l�l�l�l�l�l�l�<�H�P�T�M�H�A�<�2�/�,�/�3�4�<�<�<�<�<�<�������
����
����������¸²¯¶�������0�<�I�U�[�^�\�S�I�0�#���������������#�0��(�.�4�A�Q�Z�h�f�d�Z�M�A�4�.�(����ĚĦĨĳĶķĴĬĦĚčāĀ�{�{�~āĈčĚ�y�������������|�y�u�y�y�y�y�y�y�y�y�y�y��� �!�����������������A�M�Z�[�c�^�Z�M�A�4�(�&�'�(�4�7�A�A�A�A�����������������������r�`�V�S�V�`�r����������$�+�8�=�A�=�8��������������l�x�����������������x�m�l�g�l�l�l�l�l�làìù��üùìãàÔÓËÇÁÁÇÓÞàà�6�C�I�O�\�h�q�h�\�X�O�C�=�6�2�5�6�6�6�6������	��������������������n�y�{ŀ�}�{�r�n�b�U�T�U�b�e�n�n�n�n�n�nD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDrDsD{�4�;�@�F�F�@�4�'��!�'�.�4�4�4�4�4�4�4�4 K ; N + :  5 Z # S Y w ' . 4 . B / Y ~ a @ & n - 9 _ ] b > 3 ' S  [ ( 2 f V M F M H ( " � I G .  : : j - [ R \ n Y + : & ,    M  ~  �      o  ]  �  �  h  �  u  !    �  t  E  "    �  o  �  �  y  �    k  |  �  �  u  �  �  �  D  �  �  ]  �  ?  V  �  �  X  I  D  �  3  �  ,  �  .  �  �  ~  :  �  �  �  6  s    w  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  B<  ]  [  X  U  P  K  A  6  *    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  y  v  s  o  l  i  f  �  �  �  �  �  �  �  �  �  �  z  a  D    �  �  �  }  W  1  �  �          �  �  �  �  �  �  �  �  e  1  �  �  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  m  Z  B    �  �  N  b  d  [  J  3      �  �  �  �  Y  7  V  -  �  �    x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  e  M  p  k  e  `  [  V  Q  L  G  B  9  ,         �   �   �   �   �  �  �  �  �  �  �  �  o  X  A  )    �  �  �  �  �  �  |  r  �  �  �  �  �  �  �  �  �  �  �  �  �  k  G  %      /  M  $          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      	      !  #  !          �  �  �  �  �  �  h  G  (  4  >  F  O  X  ]  `  _  R  ?  %    �  �  �  E  �  �  W    .  <  G  Q  T  Q  M  E  =  5  %    �  �  �  U    �  K      (  9  >  C  J  V  \  ]  X  N  @  +    �  �  d     �  �  �  �        �  �  �  �  �  �  �  e  D  #    �  �  �  �  �                  	  �  �  �  �  �  O    �  _  i  y  ~  q  g  ]  S  P  P  M  H  @  6  %    �  �  �  z  P  �  z  m  _  Q  D  5  !    �  �  �  e  E  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  n  f  _  W  N  E  4  -  '        �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  Y  F  3     �    J  x  �  �  �  �  �  �  �  E  �  �  9  �    A  �   a  �  �  �  �  �  �  �  �  �  o  [  G  5  &    �  �  �  �  }  T  j  ^  A    F  u  �  �  y  c  ?  
  �  O  �  w  �  H  ~  �  �  �  ~  w  s  n  i  e  `  \  X  N  A  3  &          �  �  �  �  �  �  �  �  �  �    y  s  k  ^  Q  D  7  *    �  {  u  p  k  h  e  b  _  ]  [  Y  Z  ]  `  c  n  |  �  �  [  T  M  F  =  4  '       �  �  �  �  j  J  *    �  �  �  �  �  �  �  �  �  �  �  b  ,  �  �  �  f    �  P  �  ,   �  W  {  �    7  G  H  ?  .    �  �  �  @  �  u  �    :  V  $  C  ]  v  �  �  �  	  $  "    �  �  �  ~  D    �  d    J  F  C  ?  ;  7  4  .  '  !           �   �   �   �   �   �  l  �  �  �  �  �  �  �  �  \  +  �  �  w  "  �    t  �  �                    	    �  �  �  �  �  �  �  �  �  �  �  �  �  	      �  �  �  e  3  �  �  s  %  �  t  �  g  �  �  �  �  k  N  4    �  �  �  S    �  �  w  F  0    �  Y  D  .          �  �  �  �  s  Q  -    �  �  \  	  �     �  �  �  �  �  �  �  �  �  �  �  �  r  U  $  �  R  �  D  D  S  Z  W  O  D  4    �  �  �  �  U    �  �  ,  �  l    {  |  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �          9  *      h  �  �  �  t  X  3    �  �  I  �  �  [    �  -  0  .  &    �  �  �  �  m  A    �  �  w  D    �  �  r      �  �  �  �  �  �  m  W  A  ,       �  �  �  \    �  2  h  z  u  [  2    �  �  �  j  F    �  h  �  f  �  �   �            
       �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �    \  8    �  �  �  �  (  R  `  f  e  a  T  @  (    �  �  J  �  o  �  ^  �  �  �  �  �  �  �  u  c  L  3    �  �  �  |  <  �  v  �    �  �  �  �  �  �  �  �  �  �  �  s  ^  I  4       �  �  �  y  �  �  �  �  �  �  �  T    �  m    �  Y  �  �    w  �  �  !  !  !  !  !             �  �  �  �  �  �  �  k  I  '      �  �  �  �  �  �  �  �  �  �  �  �  �  s  p  o  m  l  �  �  �  �  �  �  �  �  �  u  U  ,     �  �  E  �  |  �  t  	�  
  

  	�  	�  	�  	N  �  �  0  �    �  �  �  A  �  �  �  �  a  S  ?  5  k  j  N  &  �  �  w  5  �  �  N  �  �  �  $  A  =    �  �  �  �  �  �  p  Y  =  �  6  �  �  A  �  �  e    �  �  �  h  B    �  �  t  ?  �  �  O  �  m  �  b  �  [  �      �  �  �  �  �  �  �  �  �  �  s  Y  :    �  �  �  x  _  X  Q  I  B  ;  4  /  +  &  "          �  �  �  �  �  �  e  H  4  %      �  �  �  �  �  �  �  |  L    �  d   �  \  �  5  �  �  �    �  �  �  �  l  �  1    i  �  �  ;  �  �  �  �  �  �  s  `  I  .    �  �  �  s  E    �  �  �  P