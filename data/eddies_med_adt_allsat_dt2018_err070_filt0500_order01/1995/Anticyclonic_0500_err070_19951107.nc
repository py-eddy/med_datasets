CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ұ ě��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�HU   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =Ƨ�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Tz�G�   max       @F������     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vu\(�     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @���          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >��u      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B44      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B46`      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�x�      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��,   max       C�h�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         %      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�HU   max       Pq/      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�|����?   max       ?�g��	k�      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >0 �      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Tz�G�   max       @F������     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vu\(�     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @�:           �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Aj   max         Aj      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�=�K]�   max       ?�f�A��     �  T   '   	      	   	            +  $      Z            	                                    \      /   U      
   0   7                  
   X      3   =   6   
   !   &         
   F                      O�6BN�0�N.�N7��O�N[�:O^��N�/P-�qP���N��P{�tN��GO���N��EN�qOhpO��M寂N���N��M�HUOp�#O�O���N~�OI�O��Ns��O�P�b�Na�2N���O�s�P��_O>�4O��O��:Og%Of��N��uO�N�[;P�WP<&O�7N���O�)OӂO���NdсN��O�r
O��O�'O�PpNhצN�%�OJݮNĬt�+��h��/��9X�#�
���
�o:�o;�o;ě�;ě�<D��<D��<T��<T��<e`B<u<u<�o<���<���<�1<�j<�j<ě�<���<���<�/<�h<�<�<�<��<��=o=+=C�=C�=C�=\)=\)=�w='�='�='�='�=49X=<j=@�=D��=ix�=u=y�#=�o=��=�hs=��T=�j=Ƨ�=Ƨ�88ABADHN[�����tg[NH8������������������������������������������������������������str{��������������{s�� �����

 �����MLN[t���������|tgXNMOOUZ[ahhhh[OOOOOOOOOfn���������������tgf)5Lt�������g[N5
#//41/#
�����)1?IE@5)����+/5;BNTVPNB5++++++++������������������������
#$'#
�������	).565455)������ #''���jn���������������thj����  ��������������������� �����������������������������������������������D>==>CHTacjpstpmaTIDyz}��������������{zyMLTZfhnz��������znUM)/1<HU]UTH</))))))))*)%)/<EHUafjjaUH<6/*����
#/;ACB?/
����yz}������������zyyyy�������������������������)<DE6)���������������������������``acdgnz������~zna``�������
��������,BN[eh]KFID5)����������������������������������������������������������������������������� #,,%#
 �����

������������# �������GGEBA>HMOTW\abeeaTHG�����
$/<LMH<#
��������
+1<HR<7/%���QQS[hmt����������h[Q���������������������������������������������� ��������)6;><:6)'*46<CFC964*XVamz�zzmdaXXXXXXXX������
#)010/#
���"),66BO[bda[SOB6+)""KGIN[gtu{}tg[RNKKKK����������������������������������������[Y[[ehlty~xthe[[[[[zyz{���������������z���� 

�����<�H�a�zÇÓàëôåàÓ�z�m�R�>�4�-�/�<���������������������������������=�@�A�@�=�0�&�+�0�2�=�=�=�=�=�=�=�=�=�=�B�O�T�Z�W�O�B�6�4�6�<�B�B�B�B�B�B�B�B�B�Y�f�r�u������������r�f�Y�Q�M�A�I�M�R�Y�-�0�:�;�F�S�_�h�c�_�S�F�:�0�-�,�-�-�-�-�������������������������������������������������������������y�������������������5�A�M�\�_�T�A�(�����տοٿ�����$�5��OāĨĲĦĢęč�t�[�6���������������#�&����������
�������
�0�U�_�g�g�a�U�0�
�������������������
�n�{łŇŋōŇ�{�n�m�l�h�n�n�n�n�n�n�n�n�s�������������������j�f�Q�R�Q�R�Z�h�s������������������������¿�������������˿T�Y�`�m�x�w�p�m�b�`�[�T�H�G�G�E�G�G�P�T�.�;�G�T�`�e�i�e�`�T�G�;�.�"�����+�.�����(�4�9�8�4������ݽֽнƽѽٽ��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��;�H�T�a�m�n�m�h�a�T�H�<�;�9�;�;�;�;�;�;���ʼԼͼʼ����������������������������������������������������������(�-�)�������ݿѿϿοѿݿ�E�E�E�E�E�FFFF
E�E�E�E�E�E�E�E�E�E�E������
��#�/�;�<�6�#���
��������������²¾¿������¿³²¬¨¦²²²²²²²²������������������������������D�D�EEE#E(E*E$EED�D�D�D�D�D�D�D�D�D��z�ÇÍÑÇÂ�z�o�n�h�n�z�{�z�x�z�z�z�z�����Ϲ���&�/�'�����ܹù�������������"�;�E�Q�I�9�"�	������������������������ûлڻܻлûûû��������������������������(�5�@�8�5�/�(�!���������������(�9�A�K�A�7������ݽнɽϽݽ��Ƨ������=�]�Y�J�0�����ƧƎ�u�i�d�hƎƧ�G�`�m�y���|�y�u�q�m�b�`�T�P�G�F�<�:�;�G�ݿ������(�$�������ݿֿǿɿͿϿ��5�A�N�]�g�s���������{�s�Z�A�5�)�3�4�1�5���������������������������������������˿"�.�;�G�G�;�5�.�(��	��޾׾Ӿξ����"�`�m�y�{�|�y�o�m�`�T�G�A�G�J�T�^�`�`�`�`���ʼּ�����ּʼ������������������������������
��
��������������ĿķĳĿ���������/�N�[�a�[�B��	�����������������[�t�g�[�N�9�5�5�)�&�����[�@�L�Y�^�r�~���������~�r�e�Y�L�@�9�5�8�@��"�.�;�?�G�J�G�B�;�.�"���	��	�
�����������*�6�7�2�*�������Ÿŷŷ�����ҽ������������������y�l�`�S�I�S�`�i�l�~����!�-�:�F�_�a�k�c�P�F�:�!����
����y���������������}�y�u�m�l�m�q�w�y�y�y�yŭŹ����ſŹŭŠŠŜŠŢŭŭŭŭŭŭŭŭEuE�E�E�E�E�E�E�E�E�E�EuEiEgE^E[EZEaEiEu����
�������
���������������������	��"�'�+�*�$�"��	������������	�	�	�	��������������ùàÇ�z�n�vÃÛàìóù�ź�����ֺɺ��������ɺֺۺ�������'�4�5�@�H�M�Y�M�@�4�'� ��������4�@�J�M�X�Y�e�b�Y�U�M�@�4�.�'�&�&�'�(�4�O�\�h�o�u�}�u�h�\�[�O�C�:�6�2�6�C�J�O�O T ( w E 1 h l A D  F " W A @ ,   9 a C = h K 8 D k .  E r B - ; H ` 4  F c e O T � a 4 ;  b M 3 C 2 ' I ( f ; K 5 b      �  d  ,  �    4  >  =  �  �  �  �  �  $  �  �    �         _  �  �  �  �  �  �  :  p    T  �  �  A  �  k  T  �  	  g    �  6    �  7  #  �  �  C  O  D  �  �  �  �  �;��
���
��j�T���o�D��<T��<o=<j>��u<T��=��`<�t�=#�
<���<�j<�=o<ě�<�<ě�<�j=#�
=L��=L��=�w=e`B=��='�=��-=�='�='�=���=�-=8Q�=u=}�=<j=D��=49X>%=P�`=�j=��`=\=Y�=��w=�{=���=�o=�\)>$�=��-=��
=��=�9X=�
=>o=�x�B��B��B��B�)B)�"B#޾B	4BAWBI�B�	BvvB�/B��B��B�=BE�B�HB�HB*8B�QB"�B44A�� B7�Bu�B��B�BV{BgB%�B�B"�B��B#i�BT"B��B��Bj�B��B�B�kB�fA�B��B��B�zB"9sB�gB,�?B�OB/�A�aB�	BN B	*B��BL|B��B�6B˸B	?B'�B��B�TB)��B#܀B	0B?�B�B��BH0B>2B�~B��B�#BAB>�B��B>B�+B"@(B46`A���B?�B��B?�BڮB~pB?�B;�BB�B":B�5B#W�B��BL�B��B�)B;�B@3B�uB�vA�~�B�*B?uB�/B"�B<WB,�dBN<B/�,A�u�B�dB��B	0B@0BOB1�B�EB��AǵcA���B
�A��\@޹�@��	A��A.�A�ˬA�A���A��A�@�AD��A�H{Ah) Ab�pA.�C���A�a�@��AX�A��TC�x�A��A��A���C�A�AȠf>���A���@��(A�z
A14B��Ah[�A���A�ܛA�&:AZ��Aiu@��|A�&2A�>�A���?�W*A_��A��Ak*@yߎAm�qA��gC��nA��A���A�#@:q!@�ݮ@ҳ�B��AȎlA��B
IA�z�@��@���A��eA�A�!�A�p�A��CA�{�A�AEP�A��Ah�NAb��A/�C��-A���@�AX��A���C�h�A��gA��@Aҡ�C�DYA�}�>��,A�^�@��A��FA0��B?�Ag�A�g�A�]CA�~�A[�Ai�g@��AA�R�A��YA�!?��A^�
A��pA��@y��An��A���C��
A�w�A�`Á�@<hA@΅�@��BE|   '   
      
   	            +  %      [            
         	                           \      0   V         1   8                  
   X      4   >   7   
   !   '            G         !                %                        1   9      /                  #                     #               %   9            =                     #      /   +         !   %                     !               !                        %         %                                       #               #   5            9         !                  /                                    !            O���N��yN.�N7��N�*�N[�:N�W N�/O���O�r�N��P��N��GOp?�Nq�BN�qO/(Ok�M寂N���N��M�HUOp�#O�O���N
q?N�.uOb=YNs��O�~�PJ��N?��N���O8kPq/O'��Oa_�O�m�N�@�O/~N~��OZ�N�[;P�WOq��N�|N���O���N�Z&O���NdсN��O=CO��O�'O�PpNhצN�%�OJݮNĬt  x  �  �  [    �  �  �  j  ?  �  	X  |  �  `  {  �  -    �  �  B  Y    T  �  "  0  �  c  	�  �  0  �  x    �  �  �  �  �  �  �  �  �  8  �  ~  T  �  n  �     �  �  D  �  �  [  3��`B��/��/��9X�t����
;��
:�o<u>0 �;ě�=��<D��<���<�o<e`B<���<�t�<�o<���<���<�1<�j<�j<ě�<�=t�=e`B<�h=\)=<j<��<��=49X=�P=C�=#�
=t�=�P=�P=�P=�7L='�='�=�7L=��=49X=@�=�\)=D��=ix�=u=��-=�o=��=�hs=��T=�j=Ƨ�=Ƨ�@=DEDGN[gt�����tg[N@������������������������������������������������������������uv{|��������������{u�� �����

 �����SU[gt}ytmgc[SSSSSSSSOOUZ[ahhhh[OOOOOOOOO~������������������~;87:BN[gt����xtg[NB;
#//41/#
�����5>A>:5)���+/5;BNTVPNB5++++++++��������������������������
!
�������	).565455)����� ���pot��������������xtp����  ��������������������� �����������������������������������������������D>==>CHTacjpstpmaTIDyz}��������������{zyMLTZfhnz��������znUM./:<HUYUPH</........//./4<HPUZ\UQH</////�����
#/6:94/
��yz}������������zyyyy�������������������������5@C>6)��������������������������``acdgnz������~zna``���������
 �������BN[afe[GAD@)���������������������������������������������������������������������������
#**&$#
����

����������������������GGEBA>HMOTW\abeeaTHG�����
$/<LMH<#
����������
!&%#
����hhnt���������thhhhhh������������������������������������������������������������)6;><:6)'*46<CFC964*XVamz�zzmdaXXXXXXXX�����
#$**&#
���"),66BO[bda[SOB6+)""KGIN[gtu{}tg[RNKKKK����������������������������������������[Y[[ehlty~xthe[[[[[zyz{���������������z���� 

�����<�H�a�zÇÓãíêçàÓÇ�z�s�V�C�8�1�<����
������������������������������=�@�A�@�=�0�&�+�0�2�=�=�=�=�=�=�=�=�=�=�B�O�T�Z�W�O�B�6�4�6�<�B�B�B�B�B�B�B�B�B�Y�f�p�r�}���������u�f�Y�U�M�C�M�N�T�Y�-�0�:�;�F�S�_�h�c�_�S�F�:�0�-�,�-�-�-�-�������������������������������������������������������������y�������������������5�A�N�R�R�K�E�:�(������������(�5�B�O�[�h�v�}āā�}�t�h�[�O�6�.�'�)�/�8�B����#�&����������
���������
�#�>�I�Q�Q�M�0��
�������������������n�{łŇŋōŇ�{�n�m�l�h�n�n�n�n�n�n�n�n�s���������������������s�f�c�X�[�\�c�s���������������������������������������˿T�Y�`�m�x�w�p�m�b�`�[�T�H�G�G�E�G�G�P�T�.�;�G�T�W�T�T�_�T�G�=�;�.�"�!���"�"�.�������"���
�����ݽнͽнֽݽ��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��;�H�T�a�m�n�m�h�a�T�H�<�;�9�;�;�;�;�;�;���ʼԼͼʼ����������������������������������������������������������(�-�)�������ݿѿϿοѿݿ�E�E�E�E�E�FFFF
E�E�E�E�E�E�E�E�E�E�E������
��#�/�;�<�6�#���
��������������²µ¿¿����¿¸²®¯¯²²²²²²²²��������������������������������D�D�D�EEEEEEEED�D�D�D�D�D�D�D�D��z�ÇÍÑÇÂ�z�o�n�h�n�z�{�z�x�z�z�z�z�����Ϲ������ ������ܹù�����������"�?�L�K�:�"�	�������������������������ûлػڻлû������������������������������(�5�@�8�5�/�(�!������������������(�-�3�(�������ݽڽԽݽ�ƚƧ������&�L�M�@�0������ƧƁ�u�g�lƚ�G�T�`�m�y�~�z�y�s�o�m�`�Y�T�S�G�>�;�D�G�ݿ���������	������ݿӿпԿ׿��A�N�[�g�s�������w�s�Z�A�5�,�4�5�2�5�7�A�����������������������������������������"�'�.�4�.�%��	����پپ����	�� �"�m�x�y�z�y�m�m�`�T�J�N�T�`�b�m�m�m�m�m�m���ʼּݼ�����޼ּʼ��������������������������
��
��������������ĿķĳĿ���������/�N�[�a�[�B��	�����������������B�N�[�g�w��{�n�g�[�N�B�<�5�,�&�)�5�9�B�Y�Y�e�n�r�r�r�n�e�Y�N�L�D�C�L�V�Y�Y�Y�Y��"�.�;�?�G�J�G�B�;�.�"���	��	�
���������*�3�0�*��������ŹŸŸ�������߽l�y���������������|�y�w�l�l�h�e�l�l�l�l��!�-�:�F�_�a�k�c�P�F�:�!����
����y���������������}�y�u�m�l�m�q�w�y�y�y�yŭŹ����ſŹŭŠŠŜŠŢŭŭŭŭŭŭŭŭE�E�E�E�E�E�E�E�E�E�E�E�EuEiEfEaEaEiEuE�����
�������
���������������������	��"�'�+�*�$�"��	������������	�	�	�	��������������ùàÇ�z�n�vÃÛàìóù�ź�����ֺɺ��������ɺֺۺ�������'�4�5�@�H�M�Y�M�@�4�'� ��������4�@�J�M�X�Y�e�b�Y�U�M�@�4�.�'�&�&�'�(�4�O�\�h�o�u�}�u�h�\�[�O�C�:�6�2�6�C�J�O�O P % w E 6 h 8 A =  F  W 9 ; , ) $ a C = h K 8 D � !  E l O 2 ; 7 d 0  I ~ U G ; � a ' 4  \ ? 3 C 2 ! I ( f ; K 5 b  �  �  �  d    �  �  4    O  �  �  �  �  �  $  3  �    �         _  �  �  �  �  �  +  �  P    �  y  v  �  �  �  �  �  �  g    �  �    �  �  #  �  �  �  O  D  �  �  �  �  �  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  Aj  W  k  v  w  l  W  9    �  �  f      �  l    �  �  �  J  �  �  �  �  �  �  �  �  �  �  �  �  �  b  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  ]  A  [  �  �  �  �  �  �  �  �  �  �  �  y  b  K  2    �  �  �        	    �  �  �  �  �  �  �  �  m  Q  5        �   �  �  �  �  �  �  �  �  �  �  �  �  }  r  f  [  P  E  9  .  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  E    �  �  �  �  �  �  �  �  ~  v  n  `  R  B  /    	  �  �  �  �  �  �    8  V  f  j  d  Z  H  ,    �  �  q  2  �    �  i  [  f  �  �  �  �  	  �  �  L  �  %  >    �  7    @  �    d  �  �  �  �  �  �  �  �  �  s  e  V  A  -       �  �  �  �  �  M  �  	  	;  	T  	R  	1  	  �  �  S    �  <  �  �  '    _  |  u  m  f  [  P  D  6  &      �  �  �  �  �  k  R  :  "  t  �  �  �  �  �  �  �  �  v  [  5    �  �  Z    �  O  �  4  A  N  \  _  \  V  R  K  B  5  "    �  �  �  �  �  �  �  {  r  i  d  ^  Y  T  O  J  E  >  2       �  �  ]     �   �  O  c  s  �  �  �  �  �  s  `  N  <  (    �  �  �  �  �  �        $  *  %               �  �  �  �  �  {  U  4                  �  �  �  �  �  �  y  c  N  9  #    �  �  �  m  X  C  ,    �  �  �  �  \  %  �  �  s  &  �  .  �  �  �  �  �  �  �  �  �  �  �  �  �  a  8    �  �  �  W  B  @  =  :  8  5  2  0  -  *  &                �   �   �  Y  K  <  0  "      �  �  �  �  �  �  w  T  -    �  �  a    �  �  �  �  f  A      �  �  �  �  b  ,  �  �  {  3  U  T  C  (    �  �  �  �  �  �  �  �  �  �  �  m  1  �  �  �  ]  w  �  �  �  �  �  �  �    2  X  J  �  �  �  �  �  �    �  �  �  	      !        �  �  �  [     �  �  Z    �  �    �  �    *  0  "     �  �  2  �  f  
�  
=  	N  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  B  8  a  G  %  @     �  �  P    �  t  C    �    1  *    	�  	�  	�  	�  	�  	�  	b  	  �  +  �  !  �  d  4  �  X  �  �  p  �  �  �  �  �  �  �  �  �  �  �  �  d  H  -    +  ]  �  �  0  "      �  �  �  �  �  �  �  �  t  b  D     �  �  b    S  �  �  �  �  �  �  �  �  z  @  �  �  P  �  �  $  �  ?  Z  h  w  w  o  c  R  4    �  �  �  �  ]    �  +  �  �  b  �          �  �  �  �  �  �  �  �  n  P  ,    �  �  K   �  �  �  �  �  �  �  �  �  �  �  n  C    �  �  w  7  �  �  |  �  �  �  �  �  �  x  ^  B  "    �  �  �  \  /  �  �  x    �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  9  �  9    r  e  o  �  w  c  N  5    �  �  �  �  �  \    �  o     �  �  �  �  �  �  �  �  �  �  �  �  �  s  `  L  :  ,          �  �  E  h  �  �  �  \  ,  �  �    
\  	�  �  �  �  �  1  �  �  q  R  <  I  V  _  T  )  �  �  �  X    �  �  ]     �  �  �  �  Y  %  �  �  �      �  �  �  V  �  �  $  �  �   �  x  A  }  �  �  �  �  �  �  �  �  �  �  �  :  �  '  T  �  �  �  �  P  �  �  �  �    1  -      �  �  t  �  ;  i  �  �  �  �  �  �  �  �  �  �  {  r  g  ]  Q  E  9  +      �  �  e  v  l  n  [  H  5  )    �  �  �  _    �  Q  �  �  k    �      *  2  ;  A  @  @  F  O  S  S  H  #  �  �  @  �  �  �  �  �  �    f  O  5    �  �  �  ^  &  �  �    C  0  S  n  `  R  D  8  +  $             �  �  �  �  �  �  �  �  �  �  �  �  �  u  Z  =    �  �  �  �  Q     �  �  �  d  5  
  
�  
�  
�  
�  
�  
�  
�  
�  
�  
r  
:  	�  	�  	'  �    )    �  �  �  �  �  �  ~  u  n  d  W  C  #  �  �  �  Q    �    ;  �  |  e  M  4      �  �  �  �  �  s  F    �  �  >  �  �  D  9  2  $    �  �  �  o  >    �  x    �  �  +  �    o  �  �  u  b  O  <  )       �  �  �  �  �  n  Q  4  	  �  �  �  �  �  �  {  Z  9    �  �  �  {  Q  &  �  �  �  �  �  �  [    �  �  �  V  #  �  �  t  -  �  �  8  �  M  �  ?  �  �  3    �  �  �  j  8  �  �  S  �  �  N  �  �  j  c  j  V  @