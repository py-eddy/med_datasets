CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��"��`B       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Pd�#       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E���R     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vE��R     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P�           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�\�           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >�$�       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��A   max       B,j�       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��K   max       B,@�       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�kD   max       C��M       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��t       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PL�       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?�#��w�l       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >O�       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E��\)     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����T    max       @vB=p��
     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_خ   max       ?�"��`A�     �  Zh                           	                                 
   6               &   U      B   "      '   t            T   !   
   D               +         <   ;                                 �                     N�$lO˅NN1�/O6)O�N�3Na��Nј�N���O�eN��NR��Oؗ�N�ܕM��P*��O��O7��O݈�NB�3PDPP|O�M�O	��O|�hO�P>_�N0�mO� �O@��O|O���Pd�#OnINp��N���PQwO���N�7BP ��N�]NEO��#N��O�ÅO��HO �PZK�O���NJ+N?��N��NO�JO�ܾOB{�Oe\N��NIoOM�O�s�N"��N�$aN�R�ND3�O+��OmNq�I��t��u�e`B�49X�49X�D���o:�o:�o;D��;�o;�o;�`B<o<#�
<49X<49X<49X<49X<49X<T��<T��<e`B<e`B<�o<�C�<�C�<�t�<��
<��
<ě�<ě�<ě�<ě�<ě�<���<���<���<�/<�/<�`B<�h<��=o=o=o=o=+=+=\)=\)=t�=#�
=#�
=0 �=49X=D��=D��=e`B=u=u=}�=�%=���=�1=�1=����������������������xuv����������������x&)+6BEDB863)&&&&&&&&��������������������4469=BORW[]^^YOKB=64����������)-)��������������������//0<FIPUWYUKIH<930//��������������������!)/5BKCB@75) LNS[gpng[NLLLLLLLLLL#<HaeaUA97/,-*##%)458BB5/)��������������������������'-*
������$)5BN_gmg[OB5)'-')/<HUafnrnljaUH</-)/6BO`krrph[IB>0������������������������	";cmnjaTH;/"��)6[glnjg[NB5)������!������Z[[ggtw�����xtrg][Z�����������������������������

�����$/<H[b`bc_XH<������������������������������������������������������������lihjmz���������zsmll������������������������ )BlphO6�����b\]ht����������tnhbb��������������������������������������&;>:6)���������
#+0650, 
���������������'-6;BO[ht��|p[O)-/1<HKIH</----------!#/0<D<50#!!!!!!!!���������������������������	������������
#HUnz���zna</#")/<INUanuqgaUH</��������������������!")5ANg�����sg[N91+!NHHJS[`gs��������tgN��������������������TQ[htvuth[TTTTTTTTTTGDGGMOR[fd\[ROGGGGGG(####'/;HTW[]_THE;1(`amz����������zmgea`]VWY^cmz�������zpma]�������
 ! 
�������')/+)"���)+)()*)
&)/3985)!��������
 !
���yxzz|������zyyyyyyyy_chhtz������th______qswz}������������}zqaaknnpswxpnmhgeaaaaa�����-)%����068<BBOPZ[ahif[OB760�������������������������������������s�i�k�s�u������ĳĿ����������������������ĿĳĩĦĥĪĳÇÎÓÞÜÓÇÄ�z�y�zÂÇÇÇÇÇÇÇÇ�������	����!��	�������������������佅�������������������������y�m�l�a�l�w��������������������ûùõù�������������žZ�c�f�l�g�f�b�Z�T�M�L�M�O�Y�Z�Z�Z�Z�Z�Z�T�a�m�n�s�o�m�e�a�T�O�H�E�F�H�R�T�T�T�T�ݽ�����ݽԽнĽ����������Ľнڽݽݻ������#�#��������ֻܻܻ�����)�+�6�@�B�E�H�B�6�0�)�����'�)�)�)�)�O�R�[�e�h�[�O�F�E�J�O�O�O�O�O�O�O�O�O�O��"�.�A�F�C�.�"���׾ʾ������ʾ׾߾���`�m�v�r�m�`�^�T�N�G�;�:�5�;�@�G�G�T�\�`�������������������~���������������������*�S�a�k�ƁƓƁ�u�h�U�C�����������*�ʾ׾�����������Ѿʾ��������������������	��!�&�#�$�%�"����	�������������������������ƾɾ¾���f�Z�M�?�>�M�T�Z�f���ùϹܹ��ܹϹǹù��ùùùùùùùùù��;�H�T�a�h�j�i�]�T�H�;�/���������
�4�;��������������ƳƫƧƪƱƳƲƪƳ���ٿĿѿ��������ݿĿ��������������������u�}ƁƎƎƐƎƁƀ�u�h�\�Q�O�N�O�X�h�q�u�����������������������������������������ѿݿ��������������ݿѿ̿ɿɿѿ���(�N�s���������������Z�5�����������������������������������������������ҹϹܹ����%�0�'����Ϲù������������ù�àìùý��������ùìàÓÇÆ�~�~ÄÇÎà�g�s�����������������z�s�g�Z�W�R�Z�c�g�g���������������������������������~��������!�����׼Ӽ��������p�g�j����ּ������
�����
������������������������/�4�<�E�@�<�/�#�!��#�$�/�/�/�/�/�/�/�/�	��"�.�;�G�Q�I�G�;�.�"���	��	�	�	�	�������������������������{�z�}�������žA�M�f�����������������|�s�A�/�#�"�2�A�����������������������������������������û���'�0�8�=�?�>�4�'�����ܻɻĻ���¥ �ּ���������ּѼԼּּּּּּּֿT�`�x�|�y�m�]�T�G�.� ������"�.�=�T�������������	����������������"� �����������������������4�M�f�s�y�v�f�Z�M�C�A�4��������4���������������������y�t�k�`�[�`�l�q�y�����	���������������g�T�N�Q�]�q���������Ŀѿݿ�����-�5�-�����ݿѿĿ������ľZ�_�f�i�f�_�`�Z�V�M�J�I�M�U�Z�Z�Z�Z�Z�Z�ѿݿ���ݿѿп˿̿ѿѿѿѿѿѿѿѿѿ��H�T�a�m�z�{�z�u�m�a�T�S�H�C�H�H�H�H�H�H�������
��#�,�0�*��
������������������Ŀ������������ĿĳĚĖĎČčĕġĦĳľĿŇŔŠŭŹ����������ŹŭŠŗŔŋŇņŅŇ���)�5�C�N�U�Z�Z�N�B�5�)���� �������������������������������������������¦���
��#�0�<�K�U�[�U�<�7�#��������������
DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDgDhDo�f�r�����z�r�i�f�`�[�`�f�f�f�f�f�f�f�f�F�S�_�_�b�_�[�S�L�F�>�?�B�C�F�F�F�F�F�F�!�-�:�F�S�_�k�l�p�l�_�S�F�>�:�-�)�!��!�ɺ˺ֺٺֺɺ������������������ƺɺɺɺɺr�~���������������������~�r�e�a�_�e�l�r���������ɺʺ˺ɺ������������������������������(�,�/�(� ��������� 5 I 5 * A a R 9 H 7 P D B � O Q * F L F = ) ? 0 . / ] v I ! 8 L b A , q % Q F I Q N # w Y M @ ` : Q L K C H , , O f h # L c t � < / H    �    T  y  O  t  �  �  �  ,  �  |  #    )  +  0  �  7  h  �  i  �  5    C  �  �  _  �  =  �  �  i  {  �  �  �  �  e  E  g  e  j  ^  �  4  I  �  �  r  �  <  $  �  �    .  +  -  J  �  >  �  �  (  ��49X<t��o<t�;��
<#�
;D��<�o<#�
<�t�<e`B<49X=o<T��<D��=\)=\)<�9X='�<��
=�+=�P=o<���=��=aG�=��`<ě�=� �=Y�=�P=}�>\)<�h=\)<�=�;d=q��=t�=\=t�=\)=,1=C�=��P=]/=<j=�j=�^5=0 �=�w=#�
=��=�7L=u=�hs=aG�=T��=��>�$�=�%=�\)=���=�1=�G�=��`=�x�B �B�0B��B�gB��B�<B��B�)B&��B!�}BVB׶B�B`B!{�B��B�7BYB~aB m.A��AB��B��B	�*B�B9�B��B`nB��B!ԹA��(B)�B��B�1B�kBTHB��B$E�B��B�B�yB%�eB!	�B�iBO�B�B,j�B	@oB	��B��B��BڜA���B �[A��\B�4B��Bc�BtB�B�BRB��BűB�4B��B͊B �>B��B�CBGfB��B>�BYB�wB&?�B!��B΄B	=�B:)B�_B!�B"�BC�Bx�B?B =�A��KB��B�+B	ƽB?�B@tB�rB� BD@B"BA���B>�BC�B��B *�B1�B��B$@B�>B�XB��B%I�B �XB})B+�B��B,@�B	��B	��B�tB��B1�A��B �1A��uBºB�BH�B@GB<B��B@�B�0B�\B@�B��B�AF4A�b�A��*A�^�A��Aΐ6A?q�A��A)X�@��&A�lA��A[��Ah%Ae�B ��AR]5A���AEX{>�kDA�8+B!iAx��B��A��A� *A���Aϱb>�t5A˱.A��A�@��A��A	Aa�aA�ƿA?��@���@��?A�nAc�Ad�`A�I�A�v�A:6�ARfA�LAb�A>ψA|5�A�_�A�~A�14A�B�A�9NA�
�A���A��C��M@�n@���@��u@(GU@x5@d�A4+CAF��A�׷A�HNA���A
A΀&A?�A�o�A*��@�j�AրA��A^�@Af��A�4A��AR��A�{�AH�S>���A�z}BSzAuoYBn�A���A�v�A�i�A�a�>�6�A���A�yA��@��A�g�A��Aa�A��OA@��@�%^@�!�A�{�A �Af��Aԍ�AҀ�A;JA�A�\A~�A>�mA} �A��aA�y5A�}A�=UA���A�o�A�pnA�cC��t@��o@��@�«@ ��@-@�=A4�_                           
                                 
   6               '   V      C   "      '   u            T   "   
   E   	            +         <   ;   	                              �      	                                                      %         /         %      '   %   !            /      '            ;            '   #      %               !   !      5   #                                 !                                                            !         )         #      !   #   !                  #            9                     !                  !      %                                                         NI�UO���N1�/N��O�N�3Na��N�?�N���N��N��NR��O�ܐNk��M��P�(O��O#��O��NB�3O݌�O���O�M�N�eO �&O��Oi��N0�mO�9 O��N���Ot�PL�OnINp��NV��OYj!O69N�7BO��N�]NEO��#N��OV��O��HN��JP�^Op8&NJ+N?��N��NO�UOr�OB{�OAqN��NIoOÕOo��N"��N�$aN�R�ND3�O��OmNq�I  �  ]  j  W  j  �    �  �  r    n  P    �  �  �  �  �  b  R  y  �      �  	@  �  	  5  �  �  2  �  �  �  	�  �  �  	)      �  �  �  �    <  V  �  �  �  y  �  "  ;  �  �  �  ]  5  7  �  �  v  q  ���C��49X�e`B���
�49X�D���o;�o:�o;�o;�o;�o<t�<t�<#�
<e`B<49X<D��<e`B<49X<�9X<u<e`B<u<�1<�=m�h<�t�<�h<���<���<�<��<ě�<ě�<���=�7L=\)<�/=�P<�`B<�h<��=o=8Q�=o=�P=49X=P�`=\)=\)=t�=H�9='�=0 �=H�9=D��=D��=u>O�=u=}�=�%=���=� �=�1=����������������������~{����������������~&)+6BEDB863)&&&&&&&&��������������������4469=BORW[]^^YOKB=64����������)-)��������������������//0<FIPUWYUKIH<930//��������������������!)/5BKCB@75) LNS[gpng[NLLLLLLLLLL #/<H_daUM=</.# ")24) �������������������������%%�����$)5BN_gmg[OB5)'.(*/6<FHUacmjhaUH</.),26BOT[hnnlh[O4 ��������������������	";Tafhe_QH;/")>N[gklkg[N5)������!������g^\^gqtu~�����tggggg���������������������������������������"&-/<HJNPPPNKH</*&#"������������������������������������������������������������jjmmz�������zvmmjjjj������������������������)Bfje6�������b\]ht����������tnhbb����������������������������������������"')))'����������
#"!
��������������)-08BO[htx{zwoiOB6*)-/1<HKIH</----------!#/0<D<50#!!!!!!!!���������������������������	������������"  #/6<HUanuvnlaU</"")/<INUanuqgaUH</��������������������@89BN[gt�������tg[N@URSW[bgt��������tg[U��������������������TQ[htvuth[TTTTTTTTTTGDGGMOR[fd\[ROGGGGGG-,,/6;@HMQTVTQIH;71-hfcmz�����������zumh]VWY^cmz�������zpma]�����

��������')/+)"���)+)()*)	")-1475)'��������

����yxzz|������zyyyyyyyy_chhtz������th______qswz}������������}zqaaknnpswxpnmhgeaaaaa�����$	����068<BBOPZ[ahif[OB760�����������������������������������s�r�s�y��������ĳĿ����������������������ĿĳĬĩĪİĳÇÎÓÞÜÓÇÄ�z�y�zÂÇÇÇÇÇÇÇÇ����	�����	�	�������������������������������������������������y�m�l�a�l�w��������������������ûùõù�������������žZ�c�f�l�g�f�b�Z�T�M�L�M�O�Y�Z�Z�Z�Z�Z�Z�a�i�m�p�m�l�a�T�H�I�T�W�a�a�a�a�a�a�a�a�ݽ�����ݽԽнĽ����������Ľнڽݽݻ�������"�"��������ܻܻܻ����)�+�6�@�B�E�H�B�6�0�)�����'�)�)�)�)�O�R�[�e�h�[�O�F�E�J�O�O�O�O�O�O�O�O�O�O��"�.�;�A�D�B�.�"�	���׾վҾ׾���	��`�m�r�p�m�`�\�T�G�C�G�J�T�^�`�`�`�`�`�`�������������������~���������������������*�O�\�e�v�~ƃƂ�u�h�\�C�*������� ��*�ʾ׾�����������Ѿʾ��������������������	���"�$�"�"�"�#�"��	�������������������������¾ƾ�������f�Z�M�E�D�M�f����ùϹܹ��ܹϹǹù��ùùùùùùùùù��;�H�T�`�d�c�W�R�M�H�;�/�"�	� �����"�;���������������������ƷƳƲƷƶư���Ŀѿ��������ݿĿ��������������������\�h�uƁƋƎƎƎƁ�~�u�h�\�S�P�[�\�\�\�\�����������������������������������������ݿ����������������ݿڿտտڿ��g�s�����������s�Z�N�A�5�(�!��%�5�A�N�g���������������������������������������ҹ��ùϹܹ����������ܹù�����������Óàìùÿ����ÿùìåàÓÌÇÁÃÇËÓ�s���������������s�g�[�Z�V�Z�g�g�s�s�s�s���������������������������������������������ݼԼм������s�i�m��������ּ������
�����
������������������������/�4�<�E�@�<�/�#�!��#�$�/�/�/�/�/�/�/�/�"�.�;�G�N�G�=�;�.�"� ��"�"�"�"�"�"�"�"�����������������������������������������4�A�M�Z�f�s�|�������s�f�Z�M�A�<�/�2�4�����������������������������������������ܻ����+�4�9�:�7�4�'�����ܻɻŻŻл�¥ �ּ���������ּѼԼּּּּּּּֿT�`�x�|�y�m�]�T�G�.� ������"�.�=�T�������������	��������������������������������������4�M�f�s�y�v�f�Z�M�C�A�4��������4�y�������������������y�s�l�i�l�w�y�y�y�y���������������������������s�b�\�^�j�}���ѿݿ��� ����������ݿѿȿ��¿ɿѾZ�_�f�i�f�_�`�Z�V�M�J�I�M�U�Z�Z�Z�Z�Z�Z�ѿݿ���ݿѿп˿̿ѿѿѿѿѿѿѿѿѿ��H�T�a�m�z�{�z�u�m�a�T�S�H�C�H�H�H�H�H�H���
���#�&�#���
��������������������ĦĳĿ����������ĿĳĦĚėďČĎĖĚĢĦŇŔŠŭŹ����������ŹŭŠŗŔŋŇņŅŇ�5�6�B�N�Q�W�V�N�B�5�)��������)�5����������������������������������������¦����#�0�<�D�I�J�<�0�#��
��������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDxD~D��f�r�����z�r�i�f�`�[�`�f�f�f�f�f�f�f�f�F�S�_�_�b�_�[�S�L�F�>�?�B�C�F�F�F�F�F�F�!�-�:�F�S�_�k�l�p�l�_�S�F�>�:�-�)�!��!�ɺ˺ֺٺֺɺ������������������ƺɺɺɺɺr�~�������������������~�r�l�e�b�`�e�n�r���������ɺʺ˺ɺ������������������������������(�,�/�(� ��������� ( D 5 % A a R 1 H 8 P D $ u O P * L K F 9 ! ? 7 % " S v = # 2 B _ A , T - F F A Q N # w ] M ? 9 - Q L K E B , / O f a   L c t � / / H    Y  �  T  �  O  t  �  �  �    �  |  �  �  )  �  0  r  �  h      �    W  U  �  �  �  9  �      i  {  k  �  �  �  �  E  g  e  j  �  �  �  �  �  �  r  �  _  �  �  W    .  �  �  J  �  >  �  T  (  �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  �  �  �  �  k  T  <  $  
   �   �  -  C  S  \  ]  V  L  :     �  �  �  j  1  �  �  �  V  �  v  j  j  j  j  c  Y  P  C  5  '    	  �  �  �  �  �  �  �  �  �  �    0  C  P  U  K  :  "    �  �  �  �  f    �    Q  j  a  c  T  >  .    �  �  �  �  p  K  &  �  �  �  �  �  �  �  �  �  v  ]  ?    �  �  �  n  5  �  �  {  8  �  �  e      �  �  �  �  �  �  �  �  �  �  �  ~  k  X  F  #  �  �  �  z  �  �  �  �  �  �  �  �  r  T  5    �  �  �  {  5  �  w  �  �  �  �  �  �  �  �  s  Y  =  !    �  �  �  �  �  I   �  i  p  j  ^  S  G  7  #    �  �  �  a  &  �  �  w  B          �  �  �  �  �  �  �  s  T  4    �  �  �  K  �  �    n  a  T  F  :  -         �  �  �  �  �  �  �  �  �  �  �  %  )  N  =  !  �  �  �  �  Y  ;       �  �  �  `  �  �  �                                        �  �  �  �  �  �  �  �  ~  w  p  l  k  j  h  g  f  e  c  b  a  p  �  �  �  �  |  g  E    �  �  �  �  �  �  l  /  �  o  #  �  �    s  h  [  P  B  2    �  �  �  U    �  w    �  n  �  �  �  �  �  �  �  �  �  �  �  }  p  [  C  $  �  �  y    �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  ,  �  ?  �   �  b  \  U  N  G  8  &      �  �  �  �  �  �  ~    u    �  �  *  K  R  N  =    �  �  �  u  a  ,  �  �  #  �    �  "  `  q  w  m  Y  A  &    �  �  �  �  �  j  O  /  �  �  �  ^  �  �  �  q  d  Z  O  ?  (    �  �  k  1  0  u  u  u  r  ^        �  �  �  �  �  �  �  �  �  �  h  :    �  �  K    �  �  �  �              �  �  �  �  �  D  �  x  �    �    ;  ^  w  �  �  �  �  `  3  �  �  i    �    -  &   �  �  1  �  �  U  �  �  	  	)  	>  	6  	  �  �  _  �  �  �  V  t  �  �  �  �  �  �  �  �  �  �  ~  u  l  b  V  I  <  )      �  	  	  	  	  �  �  �  �  o  @    �  y  
  \  �  |  7  c  �    -  4  /     	  �  �  x  2  �  �  H  �  �  @  �  �    �  �  �  �  �  �  �  �  �  �  z  Y  3    �  �  D  �  �    �  �  �  �  �  �  �  �  �  x  A    �  w  -  �  O  �  �  <    2  #  
�  
�  
�  "       
�  
�  
�  
M  	�  	<  e  `     �  �  �  �  �  �  �  �  �  �  �  �    t  e  S  A  /  %        �  �  �  �  �  �  �  �  �  �  m  S  7    �  �  �  E    �  �  �  �  �  �  �  �  �  �  �  �  q  Z  >  "     �   �   �   l  �    w  �  �  	  	;  	_  	y  	�  	  	e  	4  �  �    U  d  ;  /  f  �  �  �  �  �  �  �  �  �  �  �  b  7  �  �  2  �  �   �  �  �  �  �  }  l  [  C  )    �  �  �  �  �  �  �    s  h  �  �  	  	)  	)  	'  	$  	  �  �  �  /  �  R  �  8  �  �  @  �    	      �  �  �  �  �  �  �  �  �  p  Z  A  )    �  �                  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  \  E  0      �  �  �  �  �  y  X  9    	   �  �  �  �  �  �  �  �  �  �  �  w  j  ^  Q  E  9  ,         %  P  k  }  �  �  �  �  �  �  �  �  �  �  i    �    i  �  �  �  �  �  �  �  q  N  $  �  �  ~  5  �  �  A    �  �  �  �  �  �  �  �          �  �  �  �  �  b  ;    '  5    �  �  !  2  ;  5  $    �  �  �  �  �  z  D  �  �  �  j  �  b  �  �  $  B  R  V  O  A  ,    �  �  G  �  8  �  �  �  �  �  �  �  �  �  �  �  �  m  U  <       �  �  �  x  U  0    �  �  �  �  �  �  �  �  �  �  �  n  \  J  9  *         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    0  M  d  s  x  s  a  A    �  �  �  j  3  �  �    �  �  �  �  �  �  �  b  7    �  �  �  �  `    �  -  �  v  "         �  �  �  �  �  �  u  C    �  Y     �  \    �     "  -  7  7  +         �  �  �  �  �  F  �  Y  �  /  �  �  �  |  n  `  Q  M  Q  T  K  ?  4  *              /  >  �  w  k  _  S  J  J  J  J  J  H  B  =  7  2  ,  &         y  �  �  �  �  �  �  p  O  '  �  �  �  T    �  <  �  #  M  ~  G  �  }  �    H  \  W  7  �  �  /  �  �    ?  [  +  	�  5  4  3  2  1  0  /  .  ,  +  *  (  '  #        �  �  �  7  (      �  �  �  �  �  �  {  n  `  R  D  '    �  �  �  �  �  �  �  k  K  .    �  �  �  {  *  �  �  %  �  ^  �  a  �  �  �    a  ?    �  �  �  �  Z  ,  �  �  �  u  Y  >  "  V  u  t  p  d  M  ,    �  �  q  6  �  �  ~  ;  �  �  3    q  h  e  \  R  G  =  5  -  &    
  �  �  �  �  k  C    �  �  �  |  a  E  +  
  �  �  �  T    �  z  !  �  g    �  p