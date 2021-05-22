CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?ļj~��#      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�>   max       P�.�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���m   max       <���      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�        max       @FNz�G�     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(��    max       @v��Q�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�g�          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �Q�   max       <��
      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0��      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��N   max       B0E-      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C��1      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >8�   max       C��(      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�>   max       P�D      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?�c�	�      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �1'   max       <���      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�        max       @FL�����     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\)    max       @v}��R     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P@           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�Ġ          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:)�y��   max       ?�V�t�     �  QX            4   N   :   	      y   #                  $      .                     <      �                     
   >                        +   >                                       N/W&N�n�NH��Pb�P&��P@N��3N���P�.�P5_DO�N��|O�!N��OT��O���OAy*O�n/OP0�N1�N��P��Ok��O��P4�OOX�P7�O��Oi��N]��Nwe�Nʀ O�NN�jPe�N�N;��OP�P]vNu��OqkN�}O�I�P�jN�kKN~zN82N��M�>N3y�N^m{N�MO6:O3��N0��N��M���<���<�t��o�D����o�D���D����o���㼣�
���
��9X��/��`B���������\)�t���P��P����w�,1�0 Ž0 Ž0 Ž49X�49X�8Q�<j�@��@��L�ͽP�`�T���]/�ixսixսm�h�u�u�y�#��o��o��C���\)���������P������1�Ƨ��`B������mKNRV[agjge[NKKKKKKKKsz����������zrsssss06BFOQUOB96100000000)BJOSY[^]YOB) ������������������������������������������������������������������������z��������� ������zzt���������������tlhtrt�������������}vor��������������������)6BOR[_a^[OB6,)"ot�������tsoooooooootz�������������zvsqt)/6OV^t���swlh[OB5))�������������������������

���������������������t�������yutttttttttt��������������������)<MUn{����rU<0#  )*6CLO^immh\OC*���������������������������
'/#
�������QTamz���zyrmaTPNOOQ!)5N[t�������gB5)" !�������� ��������������
%''#
������� #/251/)##"      //9<EHUXUQMHC</.////������������������������������������������������������������������������������������������	��������������zlgdgmoz����������zzKP[g����������tg[NJK����������������������������������������##-/42300013/#6<HUanvz��znaUH=:7761BN[gk{������tgH2--1pz��������zrqpppppppDIKU`_VUIHDDDDDDDDDD|��������~||||||||||`ainz����|zna^``````������������������������������������������
"
������������������������������������������������#0<@IKMMMMI<;4#�����������������������ENP[ghg[ONNIEEEEEEEE�s�n�f�Z�S�Z�b�f�k�s�~�x�s�s�s�s�s�s�s�s�û��������������ûллܻ��ܻлûûû��U�N�Q�U�^�a�n�r�q�n�i�a�U�U�U�U�U�U�U�U�/���������"�;�H�T�a�k�s��|�m�T�;�/�r�c�Z�U�R�Y�`�r������Ǽ��˼ɼ�������r�A�9�A�Z�b�h�y�����������������������Z�A����������!�-�/�/�-�!�����������������ݿտѿʿ˿ѿݿ������������꺰���_�L�K�X�n�����:�l�x�}�x�h�c�X�-�ɺ��[�_�k�s���������Ⱦ��������׾����s�[�.�(���"�*�;�G�O�U�`�m�s�o�m�`�T�G�;�.�H�D�<�8�3�<�H�U�a�a�a�`�\�U�H�H�H�H�H�H����������������������������������������������������������������������������������������������������������������һ��������x�t���������лܻ�������߻һx�u�l�w�x�������������ûͻû����������x�Y�W�L���������'�M�Y�r�����{�r�`�_�YŭţũŹ�������������������������Źŭ¿º¾¿������������¿¿¿¿¿¿¿¿¿¿�������������������������������������������{�N�A�7�8�?�Z�s������������������������	�����������	��"�.�/�4�7�9�8�6�.�"��;�0�#��#�0�?�I�U�b�n�{ŇŒŎń�n�U�I�;���w�r�{�������¿ѿݿ��������ѿĿ���������ƽƵƶ����������������������������$��������$�=�I�b�q�r�k�f�[�I�0�$�������������	���*�6�>�D�?�6�*���Ä�z�w�x�ÇÓàìù����������ôìàÓÄ�H�E�D�H�U�a�e�n�z�{�z�n�a�U�H�H�H�H�H�H����s�m�g�e�f�g�h�s���������������������׾־̾־׾�����	��	��������޾׾�ŭţŠŔŏŇŃņŇŔŠŭŲŹž����Źŭŭìåäàßàéìîù��������ÿùìììì�f�g�r�����������������ּ����r�f��������������������������������������������'�3�4�>�3�'����������z�m�a�T�H�=�;�4�6�@�E�H�T�\�^�a�c�q�z�z�b�I�(�"�)�0�<�G�UŇŔŠţŔŜŝũşŔ�b�r�m�g�e�d�e�r�~�������~�r�r�r�r�r�r�r�rFSF=F1F$FFFFF$F1F@FJFQFVFcFjFpFoFcFSE�E�E�E�E�E�E�E�E�E�E�E�FFFFE�E�E�E乄�y�v�������ù˹Ϲܹ����ڹù��������n�X�O�K�W�a�zÃÇÓàì������ìÓÇ�z�n�������������������Ⱦľ������������������û��������ûллллûûûûûûûûû������|�����������������������������������Ŀ¿����������ĿǿѿҿֿԿѿĿĿĿĿĿ�����������$�'�$���������������������
����'�4�8�7�4�'���������EEEEE*E1E7ECEEECE=E7E2E*EEEEEEĿĽļĿ������������������������ĿĿĿĿ���s�o�n�q�s���������������������������������������������Ľнݽ���ݽνĽ������)�&�&�)�0�6�B�B�B�?�8�6�)�)�)�)�)�)�)�)�l�b�c�l�l�y�������������������y�l�l�l�l�t�t�t�|ĀāĉĈčččā�t�t�t�t�t�t�t�t A R Q ' ( o 2 ^ [ P T < T & T T 6 S f 4 ] \ W T n : 5 ! + a c K F D V  O J ; X � | c L , = J J � ? u b 7 A a ; v    T  �  w  v  �  �  �  <  	  ^  b  �  �  �  �    �  z  �  >  �  K  )  5  %  �  -  �  �  �  �  �  /  �  <  G  d  �  x  �  �  f  �  �  �  )  D  �  [  W  �  �  �  �  �    b<��
;�o��o�]/���T��O߼��
���V�]/�ě��t��#�
�\)�@���+�aG�����q���',1�}󶽅��y�#���`�e`B�Q녽�hs�}�e`B�P�`�ixսy�#�e`B��`B�Y��ixս�\)���
��%���T��hs�����#���-��+��hs���-��Q콛�㽲-��1��;d��F�����JB��B�;B;�B[�BPKB��B+��B �B�B��B��B!`�B*�B��B^�B��B ϳB#y	B�B�&B�B'�B0��B��B��A��B.�B<B��B�!B�9B]�B
BU�B-�B�]B#SRB W�B	�B"3B��BN�B��B	(�B�B'�B
��B��Bt�B)~zB�BrB��B%�.B��B6�B�hB��B��B@iBF�B�
B=dB+jB ��B�[BunB��B!@�B�dB��BE�B=B �WB#@B*�BƭBǒB(@"B0E-B�	B?�A��NB:�B(&BŌB��B;�B?�B?�B7B-=�B��B#JpA��]B	�B"�
B�rB�QB�B	CPB��B'7B
�mB��B�&B)JcB�qBӝB*�B%�fB�kBD�B��AA�x@��UA�n]A�+�@�H�A�o�@e�A��@G�RAJ��Ad��A�«A���A �EA�@�ߣ@��9@�eLA��A�4�A�[�A�]:A^�kA�dAv�TB�B
�yA��A˹�A�@�A�$�AV��A���A���@�&WA���?���A�mA�$S?���C��1C�}=��A�>�AK��@�t�A���Ax��B�J@�;C���A㴳A���A&�bAׂA$A�%�AB�8@��AƀA�vr@�|�A�w@d�sA�pq@O�:AH&Ac��A��~A�{�A �A���@��@���@ҿ�A��iA�bHA�/�A���A]AAu$�B�B
�A�?À�AŁ�A���AS(�A�xA�iA)A�?�E�A���A��?�KaC��(C��j>8�Aˆ�AK�@�T�A��bAxzB	 R@�]iC��A�t�A��NA$�LA�{jAxA�t0            4   N   :   	      y   #                  %      .                     =      �                     
   ?                        ,   >                                                   '   '   3         I   1                  )      +            -         -      +                        /            '            !   '                                                      !   /         A   )                                    -                                       +            '               '                                       N/W&N��iN7O�c9O��P�JNQ�hN�p�P�DPߕO�N��|O9!N��OT��N��POAy*OnYHOP0�N1�N��P��O^L�O��O�OOX�O)�6OY�gOi��N]��NAtNʀ N��N�jP�!N�N;��OP�P]vNu��NGW�N�}O}ґO��N�6�N~zN82N��M�>N3y�N^m{N�MO#Y�O3��N0��N��M���  �  �  �  �  	  +  �  �  
$  �  �    �  Z  �  =  w  T  �    �    X  �  	.  �  �    +  s  �  c  �  4  �  �  h  �  x  �  �  �  	  
T  D  �  �  �  S  �  �    .  �  ]  �  �<���<�C���o��9X��j��t��e`B��t��+���ͼ��
��9X��`B��`B���D�����<j�\)�t���P��P��w��w��%�0 ž1'�<j�49X�49X�<j�<j�D���@��P�`�P�`�T���]/�ixսixս�7L�u��o�����+��o��C���\)���������P������ ŽƧ��`B������mKNRV[agjge[NKKKKKKKKsz���������zstssssss46BDOPSOB>6644444444)6BGLOPNEB6)		�������������������
��������������������������������������������������������������������t���������������tnmtrt�������������}vor��������������������)6BOP[^`][OB:6-)$ot�������tsoooooooootz�������������zvsqt=BOW[^hhmjh^[SOCB:==��������������������������	
����������������������t�������yutttttttttt��������������������)<MUn{����rU<0#  )*6CKO]hlg\OC6*����������������������������������������QTamz���zyrmaTPNOOQ35=BN[]ghkg^[NB85103�����������������������
%''#
������� #/251/)##"      //1:<HHKPKHA<///////���������������������������������������������������������
���������������������������������	��������������zlgdgmoz����������zzKP[g����������tg[NJK����������������������������������������##-/42300013/#:<HUafnryxndaUH?;89:3BN[gx�������tgJ4//3rz��������zutrrrrrrrDIKU`_VUIHDDDDDDDDDD|��������~||||||||||`ainz����|zna^``````������������������������������������������
"
������������������������������������������������#0<@IKMMMMI<;4#�����������������������ENP[ghg[ONNIEEEEEEEE�s�n�f�Z�S�Z�b�f�k�s�~�x�s�s�s�s�s�s�s�s�û����������û̻лܻ��ܻлûûûûû��U�S�S�U�`�a�n�q�o�n�a�a�U�U�U�U�U�U�U�U�;�!����"�/�:�H�T�a�c�k�o�m�h�a�T�H�;��n�d�^�[�\�k�y�����������������������K�P�a�j�l�}�����������������������s�Z�K�������!�+�+�!����������������ݿؿѿοѿݿ�������������������s�[�X�c�w������!�_�l�o�_�X�H�-�ɺ����e�c�h������������׾������׾ʾ����s�e�.�(���"�*�;�G�O�U�`�m�s�o�m�`�T�G�;�.�H�D�<�8�3�<�H�U�a�a�a�`�\�U�H�H�H�H�H�H����������������������������������������������������������������������������������������������������������������껷���������������ûлһܻ�ܻڻлϻû����x�u�l�w�x�������������ûͻû����������x�M�@�4�(����'�@�M�f�o�r�u�v�r�o�f�Y�MŭţũŹ�������������������������Źŭ¿º¾¿������������¿¿¿¿¿¿¿¿¿¿�������������������������������������������{�N�A�7�8�?�Z�s������������������������	�����������	��"�.�3�6�8�8�7�4�.�"��;�0�#��#�0�?�I�U�b�n�{ŇŒŎń�n�U�I�;�����������������Ŀѿݿ޿޿ۿɿĿ�����������ƽƵƶ����������������������������0�,�$�!��"�$�/�0�=�I�V�X�[�[�V�S�I�=�0����������������*�6�<�B�<�6�*���Ä�z�w�x�ÇÓàìù����������ôìàÓÄ�H�E�D�H�U�a�e�n�z�{�z�n�a�U�H�H�H�H�H�H�������s�o�g�f�g�s�����������������������׾־̾־׾�����	��	��������޾׾�ŭťŠŔőŇńŇŇŔŠŭűŹż��żŹŭŭìåäàßàéìîù��������ÿùìììì�r�h�r�������������������ּ��r��������������������������������������������'�3�4�>�3�'����������z�m�a�T�H�=�;�4�6�@�E�H�T�\�^�a�c�q�z�z�b�I�(�"�)�0�<�G�UŇŔŠţŔŜŝũşŔ�b�r�m�g�e�d�e�r�~�������~�r�r�r�r�r�r�r�rFFFFF$F1F7F=FCF=F1F$FFFFFFFFE�E�E�E�E�E�E�E�E�E�E�E�FFFFE�E�E�E乄���{���������ùϹܹ����عù��������n�\�R�P�]�zÁÆÉÓì��������ìÓÇ�z�n�������������������ƾ¾������������������û��������ûллллûûûûûûûûû������|�����������������������������������Ŀ¿����������ĿǿѿҿֿԿѿĿĿĿĿĿ�����������$�'�$���������������������
����'�4�8�7�4�'���������EEEEE*E1E7ECEEECE=E7E2E*EEEEEEĿĽļĿ������������������������ĿĿĿĿ���s�p�o�r�s���������������������������������������������Ľнݽ���ݽνĽ������)�&�&�)�0�6�B�B�B�?�8�6�)�)�)�)�)�)�)�)�l�b�c�l�l�y�������������������y�l�l�l�l�t�t�t�|ĀāĉĈčččā�t�t�t�t�t�t�t�t A I _  7 k * \ W G T < S & T , 6 - f 4 ] \ T T R :   + a b K C D Q  O J ; X 1 | ^ N . = J J � ? u b 2 A a ; v    T  �  I      Q  X  �  �  �  b  �  Z  �  �  �  �  �  �  >  �  K  �  5  7  �  g  �  �  �  �  �    �    G  d  �  x  �  _  f  I  b  �  )  D  �  [  W  �  �  ^  �  �    b  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  �  |  o  ]  G  1       �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  R  �  ~  /  �  �  0   �  ]  {  �  �  �  �  �  r    �  �  c     �  �  I  �  �  X    I  �  �  0  ^  ~  �  �  �  �  �  l  D    �  p    n  �    v  �  �  �  	  	  	  �  �  �  :  �  x  �  B    �  \  �  �    %  +  $    �  �  �  �  �  �  O    �  ;  �  a  �  �  U  �  �  �  �  �  �  �  �  �  �  �  �  �  y  m  `  R  S  o  �  r  �  �  �  �  o  X  =    �  �  �  j  8  	  �  J  �  N   �  	�  
	  
"  
  	�  	�  	�  	S  	  �  _  �  d  �  '  �  -  p  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  B    �  u  �  m  �  �  �  �  z  n  _  O  ?  0  !       �   �   �   �   �   �   �            �  �  �  �  �  �  z  P  &  �  �  �  m  9     �  �  �  �  �  x  i  V  B  -    �  �  �    N    �  �  u  Z  X  V  S  O  L  F  ?  8  -  "    �  �  �  �  x  $   �   Y  �  �  �  �  �  �  �  {  n  a  Z  z  }  n  \  I  6     �  �  �  �  �  �  �  �  �  �  �  9  ;  0    �  �  j    �  m  7  w  q  h  ]  I  2    �  �  �  �  �  �  b  (  �  �  _  c  f  �  �    2  ?  I  R  O  7    �  �  U    �  �     9  u  X  �  �  �  �  �  `  C  0  *    �  �  <  �  �  c    �  �  d    !  '  ,  0  .  -  +  &      	  �  �  �  �  �  �  �  x  �  �  �  ~  w  p  i  a  Y  Q  H  ?  .    �  �  �  �  �  ]      �  �  �  �  �  �  k  H    �  �  q  $  �  W     �   \  N  V  N  B  2       �  �  �  j  8    �  �  N  �  }  �  *  �  �  �  m  O  :  '  �  �  �  m  6    �  �  Y    �  L  �  �  �  �  �  �  	  	*  	*  	  	   �  �  K  �  U  �  ,  9  �    �  �  �  �  �  �  �  �  �  m  P  /  
  �  �  �  R    �  �  S  [    �    _  �  �  )  d  �  �  >  �  �  �  =  P  
  �  �  �    �  �  �  �  �  �  �  `  0  �  �  �  L  
  �  s  �  +      �  �  �  k  *  �  �  B  �  �  t  O  "  �  �  R    s  e  W  H  9  (    �  �  �  �  g  ?    �  �  �  ]  -  �  |  �  �  �  z  b  J  /    �  �  �  �  v  &  �  �  @   �   �  c  Z  N  =  %    �  �  �  Y     �  �  s  ?  (  "  2  %    �  �  �  v  e  M  2    �  �  �  m  ?    �  �  u    �   �  4    
  �  �  �  �  �  �  n  Y  I  <  7  A  E  7  "  �  �  �  �  �  �  �  �  T    �  r  &  �  �  ;  �  Y  �  .  g    �  �  �  �  �  �  �  �  �  �  �  y  q  j  b  Z  R  K  C  ;  h  T  ?  +      �  �  �  �  �  �  �  �  �  �  �  t  h  ]  �  �  m  L  +    #    �  �  �  t  A    �  �  ]  0      x  U  +  <  ?  2  %    �  �  �  �  �  m  M     �  �  g  !  �  �  �  �  �  �  �  �  �  r  ^  G  0    �  �  �  �  k  K    -  B  O  A  (    ]  �  o  :  �  �  `  �  n  �  M  �  �  �  �  y  i  j  z  S    �  �  Y    �  q     �  k    �  1  �  	  	  	  	  �  �  [    �  L  �  �    �    }  �  �  ^  
=  
S  
O  
B  
+  
  	�  	�  	V  	  �  \  �  �    �    8  7  �  @  A  C  D  C  A  >  8  (    �  �  �  �  A  �  �  i  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  f  R  >  *      �  �  �  �  �  �  p  W  =  #  S  �  �  �  �  �      *  :  F  V  i  ~  �  �  �  �    S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    (  M  r  �  �  �  �  �  �  �  e  @    �  �  �  k  ,  �  �  3  �  u      �  �  �  �  �  �  �  �  �  n  \  K  <  ,    �  �  �  h  )  .  -  %    �  �  �  z  B    �  �  H    �  l    D  >  �  �  �  i  J  *  
  �  �  _    �  k    �  �  �  _     �  ]  Q  F  :  )       �  �  }  =  �  �  f    �  �  B  �  �  �  �  �  �  �  �  �  p  Z  B  *    �  �  �  �  �  �  �  �  �  �  m  U  >  %    �  �  �  �  s  V  8    �  �  �  �  