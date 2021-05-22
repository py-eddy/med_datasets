CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�`�   max       P��n       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       <�`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @F�=p��
     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vo�
=p�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P@           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�Y        max       @��            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <D��       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�bU   max       B1�0       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��4   max       B0�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >!:�   max       C�:p       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�   max       C�,�       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          P       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�`�   max       P��n       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�֡a��f   max       ?�qu�!�S       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       <ě�       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @F��z�H     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vn�Q�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P@           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�Y        max       @��            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B   max         B       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?o!-w1��   max       ?�g��	k�     �  ]      	            '         &            M   '                     	         2         "   O               #      
                  *                                 /                                    &                  O��0N��N,��Nu�6N�mSO��Oq,*N���P��O�GP��nN��PO�WPb�O7CEN�&�Nٟ.O��O���Os
qN��$Nx�O��PͣN��VO���PľP&i�O>�N���O%zN@�PANdN� �O?F�N2��OWHWN��O�W�N��PM�`�O�N|�N��O5,�Na�3O�[O���O�O}i�ONnN�w�N��ND>�N]�qN�t�Ov��N���OP!�OykO�9�N��O->O5�Nn _NZ�N�q�N�r5N�a�<�`B<�C�<u<o;��
;�o���
�#�
�D���T���T���e`B�e`B�u��o��o��C���C���C���t����
���
��1��1��9X��j��j�ě����ͼ��ͼ�����/��h�����������o�o�o�o�+�+�C��C��\)��P��P���49X�49X�49X�8Q�<j�D���D���H�9�H�9�L�ͽY��Y��y�#�}�}󶽃o��+���-���-��1	#/8<GNPH</#
�rtx��������tsmmmrrrr

#()#








tz��������zutttttttt3<HUX`abaUHB<233333346<O[gnt���{m[B6/684:BN[gtwusmg[[ONC<87:������	

����������������������������INT[t����������tg[PI���#0bx��oxybI<�����������������|xv�������683�����������5BNTB=)������[`gt�����|utnjg[UVW[CHUanpnkhca[UQHGCCCCO[_hhptutjh[OMIIKOOO��������

��������������������������
#+/0)#
������#/<?E?=<:1/#���������������������������

 �������#0Ubkqx|~nbMIF<056;BOPSTSOB>65555555��������������������'9Ohz����u\6*����O[jqto[R6)��������������������������
������������������������������������������������[g�������������tgWT[��������������������agt�������������xtca������������������������������������������������������������2BN[gt������tgO95**2��������������������������"8>?;6�����������������������fkw�������������ytmf����������������������������������������bnu{���|tnb_\YVUVXb�����


 ���������PT^acmnpmfca`TPNLLLPTamz���������zmdTPOT~������������������~���������������������)5@HKHB5+)�����	


������������������������������������������������QU^ajnpvnaUPQQQQQQQQ������������������������������������������������������������[gtw����������tec[[|���������������~{||��������������������������� �����������
#$)$#"
�������#/39;::33/#()1)	��������������������

"#((()&#

#&./0<<AFIF@<0#�����

����������������ŹŹŻ�������������������������������������	��
�	�������������׾������������������������������������������������������������޾4�)�+�2�4�A�I�M�V�Y�O�M�K�A�4�4�4�4�4�4�l�_�S�:�:�F�S�c�l�x���������»��������l���������������������ҿݿݿѿѿĿ��������g�b�[�[�X�[�g�t�x�t�m�g�g��ѿ��������Ŀѿ�������������������m�c�X�T�`�g�v�������������������������������~�c�S�>�A�X�s������������������(�'�#�(�4�6�A�M�Q�Z�d�Z�X�Q�V�M�A�4�(�(�ʼż�������������!�7�:�3�*�����ּ��Z�N�?�<�Q�Y�O�d�g�o�������������������Z����������(�5�A�N�O�N�A�?�<�5�(����������������������������������������غL�H�L�X�Y�e�q�r�r�~�������~�r�e�Y�L�L�L�N�A�<�=�D�H�U�^�g���������������|�s�g�NàÓÌ�z�n�f�\�a�zÇÓÞßó������ùìà�����������������������������������������6�3�3�3�4�6�B�O�[�]�[�O�N�K�B�B�6�6�6�6�H�C�B�H�K�U�X�a�c�a�a�]�Z�U�H�H�H�H�H�H���������������������Ľнݽ��ܽӽĽ����������������Ľ����5�?�(���ٽ��������`�]�T�H�O�T�`�m�y�~���y�o�m�`�`�`�`�`�`���`�T�A�:�;�G�T�W�m���������ĿͿɿ���������׾����������ʾ׾� ��$�*�.�4�.�"��a�\�T�=�"����"�/�H�T�g�m�{�����m�c�a�	�����������	���"�#�#�#�"����	�ܹϹϹ̹Ϲٹܹ������������ܹܹܹ������������������������&�0�*�*����� �������� ������������u�\�U�e�o¦²����������������¦�_�U�H�@�<�8�<�H�U�a�n�zÃÇÉÇ�}�z�n�_������������ �%�/�>�?�5�3����
����(�)�-�)�������������ĿĻĿ���������
������
�����������o�k�b�`�b�o�{�{ǈǎǔǟǙǔǈ�{�o�o�o�o��ӾξӾ۾������"�'�&�!��	�������*�#���*�:�C�\�h�h�\�Z�P�O�L�C�<�6�1�*�s�g�N�5�"�#�(�A�Z�s�������������������s�n�m�i�n�q�zÇÉÇÀ�z�n�n�n�n�n�n�n�n�n�������g�Y�Z�g�s�������������������������L�K�L�P�Y�e�o�r�r�~���~�|�r�e�Y�L�L�L�L���������������������������ľž¾��������X�T�Y�`�r�����������������������r�f�X�����������������������������������������N�G�A�@�A�K�N�Z�j���������������s�g�Z�N�������������������������������������t�h�h�]�a�h�m�tĀāčĒĚĜĔďčāĀ�t�h�[�O�H�O�S�d�h�tāčĚīĪĦěčā�t�h�t�l�h�e�f�l�tāčĚĦĮĮĦĥĚčā�}�t�M�I�@�8�4�3�4�@�M�Y�f�n�l�g�f�Y�M�M�M�M�l�g�_�_�W�_�i�l�l�o�u�q�l�l�l�l�l�l�l�l���������ĺɺֺ���ֺܺʺɺ������������m�i�a�]�a�f�m�z������z�m�m�m�m�m�m�m�m�����������������������������������������:�-�!����������!�-�:�W�_�v�n�_�S�F�:�����������������ûȻлܻ���ܻлû���ŭũţŠśŜŞŠŭŹ����������������źŭ�������������ùϹܹ����ܹڹϹù������������3�@�L�Y�e�r�~���������Y�@�3�D�D�D�D�D�D�D�D�D�D�EED�D�D�D�D�D�D�D�E4E*EE*E7EBECEPE\EfEiEuEvEuEiE\ETEPECE4E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���ü�������������������������������������������������
�����
�����������������лͻû������������������ûлջܻ߻ܻӻл_�X�S�F�:�9�:�:�F�S�_�l�x�~�����x�l�_�_�`�Z�S�G�<�;�G�S�`�l�y�|���y�l�i�`�`�`�` ! B < @ P F . : 4 E \ [ H F 8 @ 0 Q S  { x + p 4 q i g : . H K 0 g \  O 8 U d 6 o K E U F O m Q H 8 _ 1 t T 6 X  K 3 M } � L e b L a > I  l  �  U  �  �    �  �  �  �  �    �  �  �  �  �  �  �  �    �  &  %  �  	  r  �  �  �  �  u  ;  G  �  A  �  �  �  U  �  H  .  �  �  �  m  �  �  :  �  �  �  �  }  u  �  m  3  �  P  �  O  i  �  �  a      �:�o<t�<D��;o�o�\)�o�ě��L�ͽ�P���ě���^5�Y���/�ě��#�
�t��<j�\)��`B�+�8Q콑hs��h�0 Žixս��t��8Q��w��󶽃o�'#�
�C��@��49X�}��㽕���P�49X��P�'H�9��P�P�`�q���L�ͽy�#��^5�P�`�P�`�H�9�P�`�Y��������}󶽧�hs���
���ͽ� Ž��㽍O߽�E���9X���B��B
�BV�B��BC�B�B�1B�VB�#B
I�B%�B%B-I�Bo�B	��B�7BE0B��B�B�B�SBA�B"�'B&�fBg�B+ B1�0B�BXB�/B*0B��B
��Bj�B
�Bl{B�yB�B	$kB�bB�4B"�jBIOB!��B!�B( qBAA�bUA���B*`Ba`B��B#�OB!��B!L BE(BHxB}�B B
B+B!B��BjTB�CB��BB3
B$��B%��B~�B�@B
+�B��B��BESB�zBZjB>�B��B
G�B&/[B�kB-��B?�B	~�B��B@B�BB��B�QBՒB?9B#B'?$BA�B*�B0�B:�B��B�B��B�\B=�BA	B
�XBD<B��B<B	�#B�^B;=B"��B�\B!��B!}B(?�B@!A��4B =�B
�XB@�B�B$>TB!�ZB!!�B02B?�B=�B4�B
�SB@vB�nB�#B��BͼB;UB=�B$�/B&7�B7�A�cJA���A2ZA��KA:��@�gAtl�A���A}�bA��3A��xA:�aA\�A���A��A�]�?��JA� AA�ǖA�K�AؖgA�־A$1�A)ޔAi�AlByAX�A��AZ��?[�A��gA��A��AƍmA�i�A�b)A�|B�AYk�B ��A��LA�_A�~%?�7ALl�@��A�(�A��A���A��A��'A�6�@���@�w�@6��A���A�M8@r��@��ZA��>!:�?��hC�+�C���C�:pA�n�A��@�o�@���A�zA��A���A2R�A�{�A:�Z@�hLAs�A���A|��A��#A�<�A94�A��A��`A���A��?�oGA�r�À�A���A؂3Ał7A%3A+�Aj|AmܴAU�A���AZr0?��A���A���A��lAǅ!A���A�|>A�3B��AZ�BYA���A�KzA��C?�4WAK�@�YA� �A��nA�y4A܁�A܁�A݆�@�T@�@4�A�A�}�@zU�@��>A�]>�?��C�&#C���C�,�Aς�A�d@���@�*A|l      	            (         '            N   '      	               	         3         #   P               $                        *                                 0                                    '                     !               %         '   #   ;      5   '            #   !               1      '   /   -               /                  !      +                                                               #                                                   !      ;      -               !   !               %         )   !               -                        +                                                               #                        O��N��N,��Nu�6N�`�OA�O:SN��O�OTB�P��nN�ӦP�O��O7CEN�&�N�4fO�jO��5O7H�N��$N1�Ov��O���N��VO8�P��O��O>�N�P�O%zN@�P0XN� �O?F�N2��OWHWN��eOm#kN*�wPM�`�O�N|�N��O5,�Na�3O�[O���N�fO}i�O#�N�w�N��ND>�N]�qN�t�Ov��N���OP!�N��-O�9�N��O	��O
��Nn _NZ�N�q�N�r5N�a�    �  ]  �  �  �  �    ?    �  �  �  B  �  (  M  �  �  k  W  �  �  L  \  t  n  	    =  x  <  U  �  �  �  $  C  �  0  J  �  �    �  o  �  �  O      �  �    W  �  M  K  H    �  �  	�  	�  �  �  �  �  �  �<ě�<�C�<u<o;�o�#�
�t��T����t���9X�T���u��h��j��o��o��t���t����㼬1���
��j��j�\)��9X���ě��<j���ͼ�/������/�����������o��w�C��o�o�+�+�C��C��\)��P��P��w�49X�L�ͽ49X�8Q�<j�D���D���H�9�H�9�L�ͽ]/�Y��y�#��+��+��o��+���-���-��1	#/5<BFE</#

	rtx��������tsmmmrrrr

#()#








tz��������zutttttttt5<HUW_aUHE<355555555>BFOR[bhnrrlh[OEB=<>=BN[gqrojgg[NGB@;:==���	����������������������������������Z\cgt�����������tg`Z���#0bx��oxybI<��������������}yw�������������)//*���������)15;9)�����[`gt�����|utnjg[UVW[CHUanpnkhca[UQHGCCCCLOS[]ghotih[OMIILLLL�������������������������
 ����������
 #(*%#
������#/<?E?=<:1/#�������������������������	����������,2<IUbijqttnbUNL?:.,56;BOPSTSOB>65555555��������������������(0COhx�����u\6*��)6O[`fe]O6������������������������� �����������������������������������������������_t������������tpgYV_��������������������agt�������������xtca������������������������������������������������������������8BIN[gt������tgXNB:8��������������������������"8>?;6�����������������������fkw�������������ytmf����������������������������������������bnu{���|tnb_\YVUVXb�����


 ���������PT^acmnpmfca`TPNLLLPTamz���������zmdTPOT�����������������������������������)25>BEHBB50)	����	


������������������������������������������������QU^ajnpvnaUPQQQQQQQQ������������������������������������������������������������[gtw����������tec[[|�������������~{||||��������������������������� ������������ ��
"#'!
��!#//79880/.#()1)	��������������������

"#((()&#

#&./0<<AFIF@<0#�����

������������żżſ������������������������������������������	��
�	�������������׾������������������������������������������������������������޾4�+�,�2�4�A�M�X�N�M�I�A�4�4�4�4�4�4�4�4�_�W�S�F�F�H�S�`�l�x�����������������l�_�������������������ĿſϿͿĿ������������g�^�_�g�t�t�g�g�g�g�g�g�g�g��ݿ����������Ŀѿݿ���������������z�m�j�h�g�k�m�t�z�����������������������������~�c�S�>�A�X�s������������������(�(�$�(�4�;�A�M�U�O�T�M�A�4�(�(�(�(�(�(��ּм��������ɼ������-�/�.�������s�Z�N�H�O�X�\�g�j�s�������������������s����������(�5�A�N�O�N�A�?�<�5�(����������������������������������������غY�N�L�J�L�Y�Y�e�r�~������~�r�e�Y�Y�Y�Y�N�A�=�?�F�J�V�_�g���������������{�s�g�NàÕÎÇ�z�j�_�a�k�zÇØàì������ùìà�����������������������������������������6�3�3�3�4�6�B�O�[�]�[�O�N�K�B�B�6�6�6�6�H�E�D�H�N�U�\�a�c�a�]�[�X�U�H�H�H�H�H�H�����������������Ľнݽ߽ڽѽĽ������������������������Ľ�����$�!���нĽ����`�]�T�H�O�T�`�m�y�~���y�o�m�`�`�`�`�`�`�j�`�T�G�F�C�G�T�`�m�����������������y�j���׾ʾ����������ʾ׾�����#�(�+�"����T�H�;�/�$�"�#�#�+�/�;�H�T�a�p�u�t�m�a�T�	�����������	���"�#�#�#�"����	�ܹӹϹιϹ۹ܹ�������������ܹܹܹ������������������������&�0�*�*����� �������� ������������k�b�[�i�s¦²����������������¦�_�U�H�@�<�8�<�H�U�a�n�zÃÇÉÇ�}�z�n�_������������ �%�/�>�?�5�3����
����(�)�-�)�������������ĿĻĿ���������
������
�����������o�n�f�o�{ǁǈǋǔǞǘǔǈ�{�o�o�o�o�o�o����ܾݾ������	�� �����	�����C�>�6�1�6�A�C�O�U�W�O�F�C�C�C�C�C�C�C�C�s�g�N�5�"�#�(�A�Z�s�������������������s�n�m�i�n�q�zÇÉÇÀ�z�n�n�n�n�n�n�n�n�n�������g�Y�Z�g�s�������������������������L�K�L�P�Y�e�o�r�r�~���~�|�r�e�Y�L�L�L�L���������������������������ľž¾��������X�T�Y�`�r�����������������������r�f�X�����������������������������������������N�G�A�@�A�K�N�Z�j���������������s�g�Z�N�������������������������������������t�l�h�_�b�h�o�tāčęĒĎčā�|�t�t�t�t�h�[�O�H�O�S�d�h�tāčĚīĪĦěčā�t�h�h�h�m�t�wāĈčĚĦħĪĦĢĚčĊā�t�h�M�I�@�8�4�3�4�@�M�Y�f�n�l�g�f�Y�M�M�M�M�l�g�_�_�W�_�i�l�l�o�u�q�l�l�l�l�l�l�l�l���������ĺɺֺ���ֺܺʺɺ������������m�i�a�]�a�f�m�z������z�m�m�m�m�m�m�m�m�����������������������������������������:�-�!����������!�-�:�W�_�v�n�_�S�F�:�����������������ûȻлܻ���ܻлû���ŭũţŠśŜŞŠŭŹ����������������źŭ�������������ùϹܹܹ�ܹٹϹù����������������3�@�L�Y�e�r�~���������Y�@�3�D�D�D�D�D�D�D�D�D�D�EED�D�D�D�D�D�D�D�EiE^E\EPECE7E*E!E*E,E7ECEFEPE\EiElEpEpEiE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���ü�������������������������������������������������
�����
�����������������лͻû������������������ûлջܻ߻ܻӻл_�X�S�F�:�9�:�:�F�S�_�l�x�~�����x�l�_�_�`�Z�S�G�<�;�G�S�`�l�y�|���y�l�i�`�`�`�` $ B < @ Q &  > + F \ 7 C 4 8 @ 0 L U  { � * p 4 d b L : ( H K + g \  O . N 8 6 o K E U F O m Q E 8 Z 1 t T 6 X  K 3 8 } � I c b L a > I  �  �  U  �  �  �  �  �    �  �  �  �  �  �  �  �  �  Z  }    �  �    �  �  
  �  �  �  �  u    G  �  A  �  �    L  �  H  .  �  �  �  m  �  �  
  �  }  �  �  }  u  �  m  3  �  #  �  O  ;  d  �  a      �  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  �            �  �  �  �  �  �  �  ~  I    �  �  2   d  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  T  /    �  �  ]  V  O  I  B  ;  5  0  ,  )  %  !          �  �  �  �  �  |  x  u  o  h  b  V  F  7  !    �  �  �  �  g  (   �   �  �  �  �  �  �  �  �  w  a  J  /    �  �  �  �  �  �  �  �    3  =  >  m  �  �  �  �  �  �  r  U  /  �  �  0  �  Y  o  R  y  �  �  �  �  �  {  i  P  2    �  �  V    �  I  �  ~  �  �  �               �  �  �  �  �  �  �  �  �  �  |  �    +  ?  *    �  �  �  �  �  �  h  '  �  q  �  o  �    �  �  �  �  �            �  �  �  Z  4      �  W  �  �  �  s  F    �  �  �  �  V  "  �  �  �  r  s  Z  7     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    d  D  $  +  t  �  �  �  �  �  x  _  D  !  �  �  r    �  �    �  �  �  �  �  �  B  <  0    �  �  y    �  Q  �  |  �    w    �  �    o  ^  M  =  .  !      �  �  �  �  i  2  �  �  n  (      �  �  �  �  �  l  S  7    �  �  �  �  a  7  
   �  D  I  6  '    	  �  �  �  �  j  0  �  �  '  �  C  �  (  k  �  �  �  �  �  �  �  �  �  �  j  P  @    �  �  q    �  �  �  �  �  �  �    T  $  �  �  %  N  <    �  L  �  7  �   �  K  Y  b  i  h  a  W  E  /    �  �  �  �  v  L    �  �  �  W  K  ?  5  ,  %        �  �  �  �  q  S  -    �  �  o  �  �  �  �  �  �  _  g  n  t  v  h  V  A  0      �  �  �  �  �  �  �  �  �  {  g  P  3    �  �  ~  6  �  �  =  �    �  !  9  H  K  K  I  8    �  �  �  ]    �  @  �    A   �  \  [  Z  X  S  M  E  ;  0      �  �  �  �  �  �  v  ]  D  S  M  G  \  ^  m  s  r  k  W  6    �  �  w  %    �  I   �  I  n  h  ]  L  6      �  �  �    `  :    �  n    �   �  �  ]  �  �  �  	
  	  �  �  �  �  T    �    e  �  �  �  =          �  �  �  �  �  �  �  �  |  h  T  <  "    �  j  3  <  =  ;  2  &      �  �  �  D  �  �  Q  �  �    �    x  n  b  S  B  .    �  �  �  �  `  6    �  �  �  |  R  '  <  6  0  +  %              �  �  �  �  �  �  �      L  U  R  P  N  I  ?  .    �  �  �  x  8  �  �  %  �  H   �  �  �  �  �  �  �  �    �  �  ^  4    �  �  p  ?    �  �  �  �  �  �  �  �    j  Q  2      �  �  �  �  �    o  `  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    r  $        �  �  �  �  �  �  h  U  J  >  .    �  �  `  �  ,  ;  A  ?  9  -      �  �  �  �  �  m  E    �  v  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _    �  O  �  }  �  ,      �      *  '      �  �  �  �  �  �  �  �  �  �  J  2  #      �  �  �  �  `  7    �  �  .  �    \  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    w  o  �  �  v  X  :    �  �  �  �  �  j  F  %    �  �  �  u  Y        
    �  �  �  �  �  �  �  �  �  �  �  �  y  ]  B  �  �  �  �  �  �  �  �  �  �  �    u  j  W  9    �  �  �  o  l  e  [  O  @  *    �  �  �  �    [  1  �  �  k     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  �  �  �  �  �  �  p  S  5    �  �  �  �  q  K    �  R  �  O  G  D  ?  6  )      �  �  �  �  Z  0    �  �  b  �  /        �  �  �  �  �  �  {  ^  ?    �  �  �  s  8  �  �    �  �  �  �  �  �  j  E    �  �  {  5  �  �  @  �  �  �  J  �  �  �  �  �  �  T    
�  
  
%  	�  	  d  �  �  R  �  �  �  �  �  �  �  �  �  q  ^  K  8  %    �  �  �  �  (   �   &              �  �  �  �  �  �  �  �  ~  g  P  9  !  	  W  N  E  <  2  )         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  V  >  '    M  ?  1  #      �  �  �  �  �  }  ]  7     �   �   �   g   ;  K  J  D  7  #    �  �  �  �  z  R  #  �  �  �  �  �  �  �  H  4      �  �  �  {  O  !  �  �  �  e  5    �  �  A  �       �  �  �  �  �  �  �  �  �  m  L  #  �  �  �  k  G  #    �  �  �    l  U  ;    �  �  ~  B  �  �  D  �    c  �  �  �  l  G    �  �  �  �  �  ^    �  �  �  L    �  �  Q  	�  	[  	  �  X    �  �  b  -  �  �  a    �  �     �  �  �  	U  	{  	�  	y  	b  	b  	5  	  �  �  }  C  (  �  �  �  }  _    �  �  �  �  �  �  �  �  �  c  >    �  �  �  z  P  &  �  �  *  �  �  q  H    �  �  �  m  :      �  }     �    �    �  �  �  �  �  �  |  w  t  r  p  n  l  j  f  ]  T  K  A  8  /  �  �  �  p  `  U  W  Y  Z  O  >  (    �  �  �  �  g  7    �  �  �  �  �  �  �  �  �  |  d  M  8      �  �  �  N    �  �  �  �  �  y  f  Q  <  %    �  �  �  �  �  �  �  �  �